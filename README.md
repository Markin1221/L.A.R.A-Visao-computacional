# L.A.R.A - Visão Computacional

Detecção de produtos com YOLO (Ultralytics) e geração automática de dataset a partir de vídeos.

## Instalação

```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Gerar dataset

Grave um vídeo de **um** produto sobre fundo liso e claro, girando o produto.

```bash
python testes-b-box/popular_dataset.py caminho/video.mp4 "coca cola 2l" --preview
python testes-b-box/popular_dataset.py caminho/video_sem_produto.mp4 --fundo
```

- Classes ficam em `dataset/classes.json` e o `dataset/data.yaml` é atualizado sozinho.
- `--preview` salva cópias com o bbox em `dataset/preview/` para conferência.
- `--intervalo` (padrão 0.5 s) controla de quanto em quanto tempo um frame é salvo.
- `--fundo` gera imagens negativas (sem produto), que reduzem falsos positivos.

## Treinar

Ao importar um vídeo com `popular_dataset.py`, o treino roda automaticamente no final
(use `--sem-treino` para pular, por exemplo ao importar vários vídeos em sequência).

```bash
python testes-b-box/treinar.py --epocas 50
```

- O primeiro treino parte de `testes-yolo/yolo11n.pt`; os seguintes partem de `modelos/melhor.pt` (fine-tune).
- O melhor modelo fica em `modelos/melhor.pt`; logs e gráficos em `runs/treino/`.

## Estrutura

- `dataset/` – dataset YOLO (`images/`, `labels/`, `data.yaml`, `classes.json`)
- `testes-b-box/` – geração de bounding boxes / dataset
- `testes-yolo/` – testes de inferência (foto, vídeo, webcam)
