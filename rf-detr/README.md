# Experimento: RF-DETR x YOLO

Pasta separada pra testar o [RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow) no L.A.R.A
sem mexer no projeto principal. Usa o mesmo `dataset/` (só lê, faz uma cópia em `rf-detr/dataset/`)
e a mesma ESP32-CAM.

## Por que testar

O RF-DETR é um detector baseado em transformer (DETR + backbone DINOv2). No COCO ele tem precisão maior
que o YOLO de tamanho parecido, e costuma aprender bem com **poucas imagens**, que é o caso do L.A.R.A.
O custo: treino mais pesado e mais lento, principalmente numa GPU de 4 GB como a GTX 1650.

| Modelo | COCO AP50:95 | Licença |
|---|---|---|
| RFDETRNano | 48.4 | Apache 2.0 |
| RFDETRSmall | 53.0 | Apache 2.0 |
| RFDETRMedium | 54.7 | Apache 2.0 |
| RFDETRLarge | 56.5 | Apache 2.0 |

Na GTX 1650, comece pelo **Nano**.

## Instalação (venv separado)

Num venv próprio pra não quebrar o do projeto:

```bash
python -m venv rf-detr\venv
rf-detr\venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r rf-detr\requirements.txt
```

Se o `pip install` der erro de compilação com o Python 3.14, crie o venv com o Python 3.12
(`py -3.12 -m venv rf-detr\venv`, depois de instalar o 3.12 pelo python.org).

## Passo a passo

Sempre com o venv `rf-detr\venv` ativado, na raiz do repo.

1. **Ver se funciona (pré-treinado no COCO, sem treinar nada)**
   ```bash
   python rf-detr/testar.py --fonte webcam
   python rf-detr/testar.py --fonte 192.168.1.6      # ESP32-CAM
   ```
   `Q` fecha, `S` salva o frame em `rf-detr/capturas/`.

2. **Treinar com o dataset do projeto**
   ```bash
   python rf-detr/treinar.py --epocas 50
   ```
   Copia o dataset pro formato do RF-DETR (`rf-detr/preparar_dataset.py`), treina e salva o melhor
   em `rf-detr/modelos/melhor.pth`. Se der *CUDA out of memory*: `--batch 1 --acumular 16`.

3. **Testar o modelo treinado**
   ```bash
   python rf-detr/testar.py --fonte 192.168.1.6 --pesos rf-detr/modelos/melhor.pth
   ```

4. **Comparar com o YOLO nas mesmas imagens**
   ```bash
   python rf-detr/comparar.py
   ```
   Mostra mAP50, precisão, recall, F1 e FPS dos dois e salva em `rf-detr/resultado_comparacao.txt`.

## Como decidir

- **mAP50 / F1**: qual acerta mais. Diferença pequena (menos de ~0.05) com poucas imagens é ruído.
- **FPS**: pra ESP32 ao vivo, abaixo de ~10 FPS já fica travado.
- **Tempo de treino**: o L.A.R.A re-treina a cada vídeo novo; se o RF-DETR levar 10x mais, pesa.

Pra uma comparação justa, treine os dois com o mesmo dataset e com mais imagens (vídeos de 30 s ou mais).

## Arquivos

- `comum.py`: caminhos, criação do modelo, desenho das caixas, leitor da ESP32
- `preparar_dataset.py`: converte `dataset/` pra `rf-detr/dataset/` (train/valid/test)
- `treinar.py`: treina o RF-DETR
- `testar.py`: janela ao vivo com webcam, ESP32, vídeo ou imagem
- `comparar.py`: YOLO x RF-DETR nas imagens de validação
