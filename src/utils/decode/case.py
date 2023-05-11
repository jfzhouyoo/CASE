import torch
import torch.nn as nn
from src.utils.config import config
from src.utils.decode.beam import Beam
import torch.nn.functional as F

class Translator(object):
    """ Load with trained model and handle the beam search """

    def __init__(self, model, lang):

        self.model = model
        self.lang = lang
        self.vocab_size = lang.n_words
        self.beam_size = config.beam_size
        self.device = config.device

    def beam_search(self, src_seq, max_dec_step):
        """ Translation work in one batch """

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            """ Indicate the position of an instance in a tensor. """
            return {
                inst_idx: tensor_position
                for tensor_position, inst_idx in enumerate(inst_idx_list)
            }

        def collect_active_part(
            beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm
        ):
            """ Collect tensor parts associated to active instances. """

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(
            src_seq, encoder_db, src_enc, inst_idx_to_position_map, active_inst_idx_list
        ):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [
                inst_idx_to_position_map[k] for k in active_inst_idx_list
            ]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(
                src_seq, active_inst_idx, n_prev_active_inst, n_bm
            )
            active_src_enc = collect_active_part(
                src_enc, active_inst_idx, n_prev_active_inst, n_bm
            )

            active_encoder_db = None

            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
                active_inst_idx_list
            )

            return (
                active_src_seq,
                active_encoder_db,
                active_src_enc,
                active_inst_idx_to_position_map,
            )

        def beam_decode_step(
            inst_dec_beams,
            len_dec_seq,
            src_seq,
            enc_output,
            inst_idx_to_position_map,
            n_bm,
            enc_batch_extend_vocab,
            extra_zeros,
            mask_src,
            encoder_db,
            mask_transformer_db,
            DB_ext_vocab_batch,
            cs_enc_outputs=None,
            cs_enc_mask=None,
            concept_enc_outputs=None,
            concept_enc_mask=None,
        ):
            """ Decode and update beam status, and then return active beam idx """

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [
                    b.get_current_state() for b in inst_dec_beams if not b.done
                ]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                dec_partial_pos = torch.arange(
                    1, len_dec_seq + 1, dtype=torch.long, device=self.device
                )
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(
                    n_active_inst * n_bm, 1
                )
                return dec_partial_pos

            def predict_word(
                dec_seq,
                dec_pos,
                src_seq,
                enc_output,
                n_active_inst,
                n_bm,
                enc_batch_extend_vocab,
                extra_zeros,
                mask_src,
                encoder_db,
                mask_transformer_db,
                DB_ext_vocab_batch,
                cs_enc_outputs=None,
                cs_enc_mask=None,
                concept_enc_outputs=None,
                concept_enc_mask=None,
            ):
                ## masking
                mask_trg = dec_seq.data.eq(config.PAD_idx).unsqueeze(1)
                mask_src = torch.cat([mask_src[0].unsqueeze(0)] * mask_trg.size(0), 0)
                dec_output, attn_dist = self.model.decoder(
                    self.model.embedding(dec_seq), enc_output, (mask_src, mask_trg),
                    cs_enc_outputs=cs_enc_outputs,
                    cs_enc_mask=cs_enc_mask,
                    concept_enc_outputs=concept_enc_outputs,
                    concept_enc_mask=concept_enc_mask,
                )

                db_dist = None

                prob = self.model.generator(
                    dec_output,
                    attn_dist,
                    enc_batch_extend_vocab,
                    extra_zeros,
                    1,
                    True,
                    attn_dist_db=db_dist,
                )
                # prob = F.log_softmax(prob,dim=-1) #fix the name later
                word_prob = prob[:, -1]
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
                return word_prob

            def collect_active_inst_idx_list(
                inst_beams, word_prob, inst_idx_to_position_map
            ):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(
                        word_prob[inst_position]
                    )
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]
                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            word_prob = predict_word(
                dec_seq,
                dec_pos,
                src_seq,
                enc_output,
                n_active_inst,
                n_bm,
                enc_batch_extend_vocab,
                extra_zeros,
                mask_src,
                encoder_db,
                mask_transformer_db,
                DB_ext_vocab_batch,
                cs_enc_outputs=cs_enc_outputs,
                cs_enc_mask=cs_enc_mask,
                concept_enc_outputs=concept_enc_outputs,
                concept_enc_mask=concept_enc_mask,
            )

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map
            )

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [
                    inst_dec_beams[inst_idx].get_hypothesis(i)
                    for i in tail_idxs[:n_best]
                ]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            # -- Encode
            (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            ) = get_input_from_batch(src_seq)
            enc_vad_batch = src_seq["context_vad"]
            
            # Encode Context
            src_mask = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
            mask_emb = self.model.embedding(src_seq["mask_input"])
            src_emb = self.model.embedding(enc_batch) + mask_emb
            enc_outputs = self.model.encoder(src_emb, src_mask)  # batch_size * seq_len * 300
            
            # Affection: Encode Concept
            concept_input = src_seq["concept_batch"]
            concept_mask = concept_input.data.eq(config.PAD_idx).unsqueeze(1)
            concept_vad_batch = src_seq["concept_vad_batch"]
            mask_concept = src_seq["mask_concept"]
            concept_adj_mask = src_seq["concept_adjacency_mask_batch"]
            # mask_concept = concept_input.data.eq(config.PAD_idx).unsqueeze(1)  # real mask
            # concept_mask = self.embedding(mask_concept)  # KG_idx embedding
            concept_emb = self.model.embedding(concept_input) + self.model.embedding(mask_concept)  # KG_idx embedding
            src_concept_input_emb = torch.cat((src_emb, concept_emb), dim=1)
            src_concept_outputs = self.model.concept_graph_encoder(src_concept_input_emb, 
                                                            src_concept_input_emb,
                                                            src_concept_input_emb,
                                                            concept_adj_mask)
            src_concept_mask = torch.cat((enc_batch, concept_input), dim=1).data.eq(config.PAD_idx)
            src_concept_vad = torch.cat((enc_vad_batch, concept_vad_batch), dim=1)
            src_concept_vad = torch.softmax(src_concept_vad, dim=-1).unsqueeze(2).repeat(1, 1, config.emb_dim)
            src_concept_outputs = self.model.vad_layernorm(src_concept_vad * src_concept_outputs)
            
            # Cognition: Encode Commonsense
            bsz, uttr_num, uttr_length = src_seq["uttr_batch_concat"].size()
            uttr_batch_concat = src_seq["uttr_batch_concat"].view(bsz*uttr_num, -1)
            uttr_batch_mask = uttr_batch_concat.data.eq(config.PAD_idx).unsqueeze(1)
            
            bsz, cs_num, cs_length = src_seq["cs_batch"].size()
            cs_batch = src_seq["cs_batch"].view(bsz*cs_num, -1)
            cs_batch_mask = cs_batch.data.eq(config.PAD_idx).unsqueeze(1)
            cs_mask = src_seq["cs_mask"] # bsz, cs_num
            
            cs_adj_mask = src_seq["cs_adjacency_mask_batch"]
            
            uttr_batch_emb = self.model.embedding(uttr_batch_concat)
            uttr_batch_outputs = self.model.cognition_encoder(uttr_batch_emb, uttr_batch_mask)[:,0].view(bsz, uttr_num, -1)
            
            cs_batch_emb = self.model.embedding(cs_batch)
            cs_batch_outputs = self.model.cognition_encoder(cs_batch_emb, cs_batch_mask)[:,0].view(bsz, cs_num, -1)
            uttr_cs_outputs = torch.cat((uttr_batch_outputs, cs_batch_outputs), dim=1)
            
            assert uttr_cs_outputs.size(1) == cs_adj_mask.size(1)
            relation_emb = self.model.relation_embedding(cs_adj_mask)
            uttr_cs_graph_outputs = self.model.cs_graph_encoder(uttr_cs_outputs,
                                                    uttr_cs_outputs,
                                                    uttr_cs_outputs,
                                                    cs_adj_mask,
                                                    relation_emb)
            
            commonsense_outputs = uttr_cs_graph_outputs[:,-cs_batch_outputs.size(1):,:]
            commonsense_mask = cs_mask
            
            if self.model.dataset == "ESconv":
                # Strategy: Encode Strategy Sequence
                strategy_seqs = src_seq["strategy_seqs_batch"]
                mask_strategy = strategy_seqs.data.eq(config.PAD_idx).unsqueeze(1)
                strategy_seqs_emb = self.model.strategy_embedding(strategy_seqs)
                strategy_seqs_emb = self.model.add_position_embedding(strategy_seqs, strategy_seqs_emb)
                strategy_enc_outputs = self.model.strategy_encoder(strategy_seqs_emb, mask_strategy)
                strategy_enc_outputs = strategy_enc_outputs[:,0,:]
                
                prior_query = self.model.tanh(self.model.prior_query_linear(torch.cat((enc_outputs[:,0,:], strategy_enc_outputs), dim=-1)))
            else:
                prior_query = self.model.tanh(self.model.prior_query_linear(enc_outputs[:,0,:]))
            
            # concept
            prior_concept_enc, prior_concept_attn = self.model.concept_prior_attn(
                query = prior_query.unsqueeze(1), # enc_outputs[:,0,:].unsqueeze(1),
                memory = self.model.tanh(src_concept_outputs),
                mask = src_concept_mask
            )
            prior_concept_attn = prior_concept_attn.squeeze(1)
            
            # commonsense
            prior_cs_enc, prior_cs_attn = self.model.cs_prior_attn(
                query = prior_query.unsqueeze(1), # enc_outputs[:,0,:].unsqueeze(1),
                memory = self.model.tanh(commonsense_outputs),
                mask = commonsense_mask.eq(0)
            )
            prior_cs_attn = prior_cs_attn.squeeze(1)
            cs_enc = prior_cs_enc.squeeze(1)
            concept_enc = prior_concept_enc.squeeze(1)
            
            # Fine-grained MIM
            bsz, react_uttr_num, _ = src_seq["react_batch"].size()
            assert uttr_num == react_uttr_num
            react_batch = src_seq["react_batch"].view(bsz*react_uttr_num, -1)
            react_batch_mask = react_batch.data.eq(config.PAD_idx).unsqueeze(1)
            react_emb = self.model.embedding(react_batch)
            react_batch_outputs = self.model.react_encoder(react_emb, react_batch_mask)
            # react_batch_enc, _ = self.model.react_selfattn(react_batch_outputs, react_batch_mask.squeeze(1))
            react_batch_enc = torch.mean(react_batch_outputs, dim=1)
            react_batch_enc = react_batch_enc.view(bsz, react_uttr_num, -1)
            react_batch_enc = self.model.react_ctx_encoder(torch.cat((react_batch_enc.unsqueeze(2).repeat(1, 1, enc_outputs.size(1), 1),
                                                        enc_outputs.unsqueeze(1).repeat(1, react_uttr_num, 1, 1)), 
                                                        dim=-1).view(bsz*react_uttr_num, enc_outputs.size(1), -1), 
                                                        src_mask.unsqueeze(1).repeat(1, react_uttr_num, 1, 1).view(bsz*react_uttr_num, -1,enc_outputs.size(1)))
            react_batch_enc = react_batch_enc[:,0,:].view(bsz, react_uttr_num, -1)
            react_batch_enc = self.model.react_linear(react_batch_enc)
            bsz, _, emb = react_batch_enc.size()
            uttr_emotion = react_batch_enc[:,0].unsqueeze(1).repeat(1, src_seq["max_uttr_cs_num"], 1)
            split_intent_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, src_seq["split_intent_num"], 1).view(bsz, -1, emb)
            split_need_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, src_seq["split_need_num"], 1).view(bsz, -1, emb)
            split_want_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, src_seq["split_want_num"], 1).view(bsz, -1, emb)
            split_effect_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, src_seq["split_effect_num"], 1).view(bsz, -1, emb)
            react_enc = torch.cat((uttr_emotion, 
                                split_intent_emotion,
                                split_need_emotion,
                                split_want_emotion,
                                split_effect_emotion), dim=1)
            assert react_enc.size(1) == cs_num
    
            # uttr_split_react_mask = src_seq["uttr_split_react_mask"]
            # fine_mask = torch.cat((torch.ones((bsz, 1)).long().to(config.device), uttr_split_react_mask), dim=1)
            # assert fine_mask.size(1) == react_batch_enc.size(1)
            # fine_emotion, _ = self.model.fine_emotion_selfattn(react_batch_enc, fine_mask.eq(0))
            
            fine_emotion = react_batch_enc[:, 0]
            emotion_emb = self.model.emotion_norm(torch.cat((concept_enc, fine_emotion), dim=-1))
            emo_gate = F.sigmoid(self.model.emotion_gate(emotion_emb))
            emotion_enc = emo_gate * concept_enc + (1 - emo_gate) * fine_emotion
            
            # Merge Context, Cognition-Affection-Strategy Signals
            if self.model.dataset == "ESConv":
                ctx_enc_outputs = self.model.ctx_merge_lin(torch.cat((
                    enc_outputs, 
                    cs_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                    concept_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                    strategy_enc_outputs.unsqueeze(1).repeat(1, enc_outputs.size(1), 1)
                ), dim=2))
            else:
                ctx_enc_outputs = self.model.ctx_merge_lin(torch.cat((
                    enc_outputs, 
                    cs_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                    emotion_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                ), dim=2))

            src_enc = ctx_enc_outputs
            mask_src = src_mask

            encoder_db = None

            mask_transformer_db = None
            DB_ext_vocab_batch = None

            # -- Repeat data for beam search
            n_bm = self.beam_size
            n_inst, len_s, d_h = src_enc.size()
            src_seq = enc_batch.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)

            # -- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
                active_inst_idx_list
            )

            # -- Decode
            for len_dec_seq in range(1, max_dec_step + 1):

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams,
                    len_dec_seq,
                    src_seq,
                    src_enc,
                    inst_idx_to_position_map,
                    n_bm,
                    enc_batch_extend_vocab,
                    extra_zeros,
                    mask_src,
                    encoder_db,
                    mask_transformer_db,
                    DB_ext_vocab_batch,
                    cs_enc_outputs=commonsense_outputs,
                    cs_enc_mask=commonsense_mask.eq(0).unsqueeze(1),
                    concept_enc_outputs=src_concept_outputs,
                    concept_enc_mask=src_concept_mask.unsqueeze(1),
                )

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                (
                    src_seq,
                    encoder_db,
                    src_enc,
                    inst_idx_to_position_map,
                ) = collate_active_info(
                    src_seq,
                    encoder_db,
                    src_enc,
                    inst_idx_to_position_map,
                    active_inst_idx_list,
                )

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)

        ret_sentences = []
        for d in batch_hyp:
            ret_sentences.append(
                " ".join([self.model.vocab.index2word[idx] for idx in d[0]]).replace(
                    "EOS", ""
                )
            )

        return ret_sentences  # , batch_scores


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = seq_range_expand
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.to(config.device)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


def get_input_from_batch(batch):
    enc_batch = batch["input_batch"]
    enc_lens = batch["input_lengths"]
    batch_size, max_enc_len = enc_batch.size()
    assert enc_lens.size(0) == batch_size

    enc_padding_mask = sequence_mask(enc_lens, max_len=max_enc_len).float()

    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        enc_batch_extend_vocab = batch["input_ext_vocab_batch"]
        # max_art_oovs is the max over all the article oov list in the batch
        if batch["max_art_oovs"] > 0:
            extra_zeros = torch.zeros((batch_size, batch["max_art_oovs"]))

    c_t_1 = torch.zeros((batch_size, 2 * config.hidden_dim))

    coverage = None
    if config.is_coverage:
        coverage = torch.zeros(enc_batch.size()).to(config.device)

    if enc_batch_extend_vocab is not None:
        enc_batch_extend_vocab.to(config.device)
    if extra_zeros is not None:
        extra_zeros.to(config.device)
    c_t_1.to(config.device)

    return (
        enc_batch,
        enc_padding_mask,
        enc_lens,
        enc_batch_extend_vocab,
        extra_zeros,
        c_t_1,
        coverage,
    )
