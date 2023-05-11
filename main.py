from tqdm import tqdm
from copy import deepcopy
from tensorboardX import SummaryWriter
from torch.nn.init import xavier_uniform_

from src.utils.config import config
from src.utils.common import set_seed
from src.models.CASE.model import CASE
from src.utils.data.loader import prepare_data_seq
from src.models.common import evaluate, count_parameters, make_infinite

def make_model(vocab, emo_num, strategy_num):
    is_eval = config.test
    if config.model == "case":
        model = CASE(
            vocab,
            emotion_num=emo_num,
            strategy_num=strategy_num,
            is_eval=is_eval,
            model_file_path=config.model_file_path if is_eval else None,
        )
    model.to(config.device)
    
    # Intialization
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)

    print("# PARAMETERS", count_parameters(model))

    return model

def pretrain(model, train_set):
    pretrain_epoch = config.pretrain_epoch
    check_iter = 200
    try:
        model.train()
        writer = SummaryWriter(log_dir=config.save_path)
        weights_best = deepcopy(model.state_dict())
        data_iter = make_infinite(train_set)
        steps = len(train_set)
        for epoch in range(pretrain_epoch):
            for n_iter in tqdm(range(steps)):
                bow_loss = model.train_one_batch(next(data_iter), n_iter)
                writer.add_scalars("bow_loss", {"loss_train": bow_loss}, n_iter)
                if config.noam:
                    writer.add_scalars(
                        "lr", {"learning_rata": model.optimizer._rate}, n_iter
                    )
            weights_best = deepcopy(model.state_dict())
    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")
        model.save_model(0, 0)
        weights_best = deepcopy(model.state_dict())

    return weights_best


def train(model, train_set, dev_set):
    check_iter = 2000
    iters = 13000 if config.dataset=="ED" else 6000
    # check_iter = 1
    try:
        model.train()
        best_ppl = 1000
        patient = 0
        writer = SummaryWriter(log_dir=config.save_path)
        weights_best = deepcopy(model.state_dict())
        data_iter = make_infinite(train_set)
        for n_iter in tqdm(range(1000000)):
            bow_loss, kl_loss, mim_loss, ctx_loss, ppl, str_loss, str_acc, emo_loss, emo_acc = model.train_one_batch(next(data_iter), n_iter)
            writer.add_scalars("bow_loss", {"loss_train": bow_loss}, n_iter)
            writer.add_scalars("kl_loss", {"loss_train": kl_loss}, n_iter)
            writer.add_scalars("mim_loss", {"loss_train": mim_loss}, n_iter)
            writer.add_scalars("ctx_loss", {"loss_train": ctx_loss}, n_iter)
            writer.add_scalars("ppl", {"ppl_train": ppl}, n_iter)
            if config.dataset == "ESConv":
                writer.add_scalars("str_loss", {"loss_train": str_loss}, n_iter)
                writer.add_scalars("str_acc", {"str_acc_train": str_acc}, n_iter)
            else:
                writer.add_scalars("emo_loss", {"loss_train": emo_loss}, n_iter)
                writer.add_scalars("emo_acc", {"emo_acc_train": emo_acc}, n_iter)
            if config.noam:
                writer.add_scalars(
                    "lr", {"learning_rata": model.optimizer._rate}, n_iter
                )

            if (n_iter + 1) % check_iter == 0:
                model.eval()
                model.epoch = n_iter
                bow_loss_val, kl_loss_val, mim_loss_val, ctx_loss_val, ppl_val, str_loss_val, str_acc_val, emo_loss_val, emo_acc_val, _ = evaluate(
                    model, dev_set, ty="valid", max_dec_step=50
                )
                writer.add_scalars("bow_loss", {"bow_loss_valid": bow_loss_val}, n_iter)
                writer.add_scalars("kl_loss", {"kl_loss_valid": kl_loss_val}, n_iter)
                writer.add_scalars("mim_loss", {"mim_loss_valid": mim_loss_val}, n_iter)
                writer.add_scalars("ctx_loss", {"ctx_loss_valid": ctx_loss_val}, n_iter)
                writer.add_scalars("ppl", {"ppl_valid": ppl_val}, n_iter)
                if config.dataset == "ESConv":
                    writer.add_scalars("str_loss", {"str_loss_valid": str_loss_val}, n_iter)
                    writer.add_scalars("str_acc", {"str_acc_valid": str_acc_val}, n_iter)
                else:
                    writer.add_scalars("emo_loss", {"emo_loss_valid": emo_loss_val}, n_iter)
                    writer.add_scalars("emo_acc", {"emo_acc_valid": emo_acc_val}, n_iter)
                model.train()
                if n_iter < iters:
                    continue
                if ppl_val <= best_ppl:
                    best_ppl = ppl_val
                    patient = 0
                    model.save_model(best_ppl, n_iter)
                    weights_best = deepcopy(model.state_dict())
                else:
                    patient += 1
                if patient > 2:
                    break

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")
        model.save_model(best_ppl, n_iter)
        weights_best = deepcopy(model.state_dict())

    return weights_best

def test(model, test_set):
    model.eval()
    model.is_eval = True
    bow_loss_test, kl_loss_test, mim_loss_test, ctx_loss_test, ppl_test, str_loss_test, str_acc_test, emo_loss_test, emo_acc_test, results = evaluate(
        model, test_set, ty="test", max_dec_step=50
    )
    file_summary = config.save_path + "/results.txt"
    with open(file_summary, "w") as f:
        f.write("EVAL\tBOW_Loss\tKL_Loss\tMIM_Loss\tCTX_Loss\tPPL\tSTR_loss\tSTR_acc\tEMO_loss\tEMO_acc\n")
        f.write(
            "{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
                bow_loss_test, kl_loss_test, mim_loss_test, ctx_loss_test, ppl_test, str_loss_test, str_acc_test, emo_loss_test, emo_acc_test
            )
        )
        for r in results:
            f.write(r)

def main():
    set_seed()  # for reproducibility

    train_set, dev_set, test_set, vocab, emo_num, strategy_num = prepare_data_seq(
        batch_size=config.batch_size
    )

    model = make_model(vocab, emo_num, strategy_num)

    if config.test:
        test(model, test_set)
    else:
        if config.pretrain:
            weights_best = pretrain(model, train_set)
            model.load_state_dict({name: weights_best[name] for name in weights_best})
            config.pretrain = False
        weights_best = train(model, train_set, dev_set)
        model.epoch = 1
        model.load_state_dict({name: weights_best[name] for name in weights_best})
        test(model, test_set)

if __name__ == '__main__':
    # prepare_data_seq(batch_size=config.batch_size)
    main()