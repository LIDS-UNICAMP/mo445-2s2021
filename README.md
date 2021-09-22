# Tarefa MO445

# Requisitos

## FLIM

Primeiro baixe o repositório do [FLIM](https://github.com/LIDS-UNICAMP/FLIM/tree/mo445) na branch mo445 e installe o pacote:


```
  cd <git_dir>
```

caso use ssh faça:
```
  git clone git@github.com:LIDS-UNICAMP/FLIM.git
  cd FLIM
  pip install -r requirements.txt
  pip install .
```

com o pacote do FLIM é possível utilizar a ferramenta de anotação com

```
    annotation <image_path>
```

## Requisitos próprios

Para instalar os requisitos próprios desse projeto faça:


```
    cd <tarefa_mo445>
    pip install -r requirements.txt
```


# Fluxo de utilização


1. Obter marcadores com a ferramenta de anotação, como por exemplo

```
    annotation imgs_and_markers/9.png
```

2. Treinar o encoder a partir das imagens com marcadores

```
    python src/train_encoder.py -a arch-unet.json -i imgs_and_markers/ -o encoder.pt
```

2.5. As ativações dos encoders podem ser visualizadas com o script [extract_features.py](src/extract_features.py), esse script irá extrair todas as features na pasta especificada:

```
    python src/extract_features.py -a arch-unet.json -i imgs_and_markers/9.png -o features/ -m encoder.pt

```


3. Executar em C programa da [libmo445](libmo445.tar.bz2) para segmentar uma imagem a partir dos marcadores e gerando uma máscara. Cada máscara deve ser nomeada como '''<original_img>_label.png'''.


4. A partir das segmentações geradas com o passo 3, treinar um modelo de rede neural com o script [train_flim_unet.py](src/train_flim_unet.py), por exemplo:

```
    python src/train_flim_unet.py -a arch-unet.json -id imgs_and_markers/ -gd gts/
```

***Obs.: Assegure que toda imagem _label.png tenha sua imagem original*** 

5. O script [eval_unet](src/eval_unet.py) avalia o modelo treinado no passo anterior, verificando a acurácia e gerando a máscara na pasta de saída passada como argumento:

```
    python src/eval_unet.py -a arch-unet.json -id imgs_and_markers/ -gd gts/ -o output/
```

