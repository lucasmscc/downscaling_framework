# Framework

Framework de Machine Learning para Downscaling Estatístico de Variáveis Meteorológicas. Desenvolvido para modelagem e previsão espacial/temporal (com foco atual na velocidade do vento e expansível para outras variáveis) utilizando dados do Global System Forecast (GFS) e dados observados de estações de superfície do INMET.

## Guia de Instalação e Configuração

Siga os passos abaixo para clonar o repositório, configurar o ambiente com o `uv` e baixar os dados necessários para execução dos testes.

```bash
git clone git@github.com:lucasmscc/downscaling_framework.git
cd downscaling_framework
```

### Configurar o Ambiente uv

``` 
uv sync
```

### Ativar o Ambiente do Projeto

``` 
source .venv/bin/activate
```

## Download dos Dados

1. Baixe o arquivo compactado da pasta de dados através do link:
    https://drive.google.com/file/d/1ajMdTZiFtXIQVXfJh47FKOqUeonI9b_o/view?usp=drive_link
2. Extraia o conteúdo para a raiz do repositório de forma que a pasta tests/data/ seja estruturada corretamente.

## Configurações e Parametrizações

O diretório de configurações (tests/data/params/config/) contém os arquivos principais que parametrizam a execução dos modelos:

1. gfs_config.json: Contém as configurações técnicas para a leitura, tratamento e utilização dos dados do modelo numérico GFS.
2. station_locations.json: Mapeia os locais e metadados geográficos das estações meteorológicas disponíveis.

## Execução
O framework possui três etapas principais: treinamento, operação e avaliação.

### Treinamento

```
pytest tests/test_train.py
```

### Operação
```
pytest tests/test_predict.py
```

### Avaliação
```
python tests/plot.py 
```