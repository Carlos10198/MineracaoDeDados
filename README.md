
# Projeto Mineração de Dados - Classificação de Carros

## Descrição do Projeto

Este projeto realiza a aplicação de técnicas de mineração de dados para classificação de carros com base em atributos presentes no dataset. O objetivo é treinar um modelo de machine learning que possa prever categorias específicas relacionadas aos carros, utilizando Python e bibliotecas comuns sobre de dados.

## Estrutura do Projeto

```
MineracaoDeDados/
├── data/
│   └── data.csv                                                        # Dataset original com os dados dos carros
├── docs/
│   └── TrabalhoFinal_Carlos_Eduardo_Bueno_103401_2023_1.pdf            # Relatório completo com análise e resultados do projeto
├── image/
│   └── matriz_confusao.png                                             # Imagem da matriz de confusão gerada pelo modelo
│   └── resultadoTerminal.png                                           # Imagem dos resultados do terminal
├── main.py                                                             # Script principal para processamento, treino e avaliação
├── README.md                                                           # Documentação do projeto
└── requirements.txt                                                    # Lista de dependências Python necessárias
```

O dataset utilizado já está incluso neste repositório:  
- 📂 `data/data.csv`

Fonte original: [Kaggle - Car Features and MSRP](https://www.kaggle.com/datasets/CooperUnion/cardataset)


## O que o código faz (`main.py`)

- **Carrega os dados** do arquivo `data/data.csv`.
- **Realiza pré-processamento**, como limpeza e preparação dos dados para o modelo.
- **Divide os dados** em conjuntos de treino e teste.
- **Treina um modelo de classificação** com os dados de treino.
- **Avalia o modelo** utilizando o conjunto de teste, gerando métricas como acurácia e matriz de confusão.
- **Salva a matriz de confusão** em `image/matriz_confusao.png` para facilitar a visualização dos resultados.

## Resultados

- O arquivo `image/matriz_confusao.png` apresenta visualmente o desempenho do modelo.
- O arquivo `image/resultadoTerminal.png` apresenta visualmente o desempenho com informações do terminal.
- O relatório `docs/TrabalhoFinal_Carlos_Eduardo_Bueno_103401_2023_1.pdf` traz uma análise detalhada, com gráficos, discussão dos resultados e conclusões.

## Como usar e rodar o projeto

### Pré-requisitos

- Python 3.7 ou superior
- Instalar as dependências listadas no `requirements.txt`

### Instalação das dependências

Execute no terminal:

```bash 
   pip install -r requirements.txt
```

### Executando o projeto

1. Certifique-se que o dataset está em `data/data.csv`.
2. No terminal, navegue até a pasta do projeto:

```bash
   cd MineracaoDeDados
```

3. Execute o script principal:

```bash
   python main.py
```

4. Após a execução, os resultados serão exibidos no terminal e a matriz de confusão será salva em `image/matriz_confusao.png`.

---

Se quiser entender melhor os resultados e análises, consulte o relatório em `docs/relatorio.pdf`.

**Autor:** [Carlos Eduardo Bueno]  
**Data:** [05/07/2025]
