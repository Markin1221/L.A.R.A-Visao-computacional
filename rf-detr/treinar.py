"""
Treina o RF-DETR com o mesmo dataset do YOLO.

Uso:
    python rf-detr/treinar.py                     # nano, 50 epocas
    python rf-detr/treinar.py --tamanho small --epocas 100

Precisa das dependencias de treino:  pip install "rfdetr[train]"

Na GTX 1650 (4 GB) use o nano: batch 2 x acumulacao 8 = batch efetivo 16.
Se faltar memoria (CUDA out of memory), rode com --batch 1 --acumular 16.
O melhor checkpoint e copiado pra rf-detr/modelos/melhor.pth.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

from comum import DATASET_RFDETR, MELHOR_MODELO, PASTA, PASTA_MODELOS, PASTA_RUNS, TAMANHOS, criarModelo


def ligarProgresso(total):
    """
    Faz o treino imprimir '  N/TOTAL  rf-detr' no comeco de cada epoca (mesmo formato do YOLO,
    e assim que a interface calcula o tempo restante) e o mAP50 no fim de cada validacao.
    O RF-DETR 1.x monta o Trainer do PyTorch Lightning por dentro, entao embrulhamos essa funcao.
    """
    try:
        import rfdetr.training as treino
        from pytorch_lightning import Callback
    except ImportError:
        return

    class Progresso(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            print(f"{trainer.current_epoch + 1:>8}/{total}  rf-detr", flush=True)

        def on_validation_end(self, trainer, pl_module):
            m = trainer.callback_metrics
            valor = m.get("val/mAP_50", m.get("val/ema_mAP_50"))
            if valor is not None:
                print(f"         mAP50 = {float(valor):.3f}", flush=True)

    original = treino.build_trainer

    def comProgresso(*args, **kwargs):
        trainer = original(*args, **kwargs)
        trainer.callbacks.append(Progresso())
        return trainer

    treino.build_trainer = comProgresso


def main():
    parser = argparse.ArgumentParser(description="Treina o RF-DETR com o dataset do projeto.")
    parser.add_argument("--tamanho", choices=TAMANHOS, default="nano")
    parser.add_argument("--epocas", type=int, default=50)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--acumular", type=int, default=8, help="grad_accum_steps (batch efetivo = batch x acumular)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--paciencia", type=int, default=15, help="para se nao melhorar por N epocas")
    parser.add_argument("--resolucao", type=int, help="multiplo de 32 (ex: 320, 384, 448). Padrao: o do modelo")
    parser.add_argument("--sem-preparar", action="store_true", help="nao recopia o dataset antes de treinar")
    args = parser.parse_args()

    if not args.sem_preparar or not os.path.isdir(DATASET_RFDETR):
        subprocess.run([sys.executable, os.path.join(PASTA, "preparar_dataset.py")], check=True)

    try:
        import pytorch_lightning  # noqa: F401
    except ImportError:
        raise SystemExit('faltam as dependencias de treino do RF-DETR. Rode:  pip install "rfdetr[train]"')

    import torch
    gpu = torch.cuda.is_available()
    print(f"RF-DETR {args.tamanho} | {'GPU: ' + torch.cuda.get_device_name(0) if gpu else 'CPU (vai ser lento)'}", flush=True)

    saida = os.path.join(PASTA_RUNS, time.strftime(f"{args.tamanho}_%Y%m%d_%H%M%S"))
    opcoes_modelo = {"gradient_checkpointing": gpu}   # economiza memoria na GPU de 4 GB
    if args.resolucao:
        opcoes_modelo["resolution"] = args.resolucao

    modelo = criarModelo(args.tamanho, **opcoes_modelo)
    ligarProgresso(args.epocas)
    inicio = time.time()
    modelo.train(
        dataset_dir=DATASET_RFDETR,
        output_dir=saida,
        epochs=args.epocas,
        batch_size=args.batch,
        grad_accum_steps=args.acumular,
        lr=args.lr,
        device="cuda" if gpu else "cpu",
        early_stopping=True,
        early_stopping_patience=args.paciencia,
        tensorboard=False,
        progress_bar="tqdm",
    )
    print(f"treino levou {(time.time() - inicio) / 60:.1f} min", flush=True)

    melhor = os.path.join(saida, "checkpoint_best_total.pth")
    if not os.path.exists(melhor):
        raise SystemExit(f"nao achei {melhor}; veja os arquivos em {saida}")
    os.makedirs(PASTA_MODELOS, exist_ok=True)
    shutil.copyfile(melhor, MELHOR_MODELO)
    print(f"modelo salvo em {MELHOR_MODELO}", flush=True)


if __name__ == "__main__":   # necessario no Windows (os workers do dataloader reimportam o script)
    main()
