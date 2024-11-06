# NLP-APS-2

# Etapa 1: Encontrando embeddings

## Descrição do Dataset

O dataset possui duas colunas, "title" e "content". A coluna "title" representa o título ou nome da vaga de desenvolvedor de software. A coluna "content" possui todas as informações, requisitos, benefícios e o que for relevante sobre a vaga em questão.

## Processo de geração dos embeddings

Foi importado o modelo e o tokenizador do BERT do PyTorch. Foi feita uma função que gera os embeddings com base no tokenizador. Para cada conteúdo, é gerado um embedding. Cada embedding gerado no BERT passa 4 vezes pelo Autodecoder para ter suas dimensões reduzidas e possivelmente ficarem mais relevantes.

A topologia da rede neural pode ser vista abaixo:
![Topologia](BERT.png)

Os hiperparâmetros utilizados foram dimensão dos embeddings, autoencoder, critério e otimizador.

## Processo de treinamento

Foi utilizado no autocoder um encoder sequencial de 3 camadas: nn.Linear(embedding_dim, 128), nn.ReLU() e nn.Linear(128, 64), além de um decoder sequencial de 3 camadas também nn.Linear(64, 128), nn.ReLU() e nn.Linear(128, embedding_dim). O número 128 é para quantas camadas quero que o embedding seja transformado, ao invés do padrão 768 do BERT. Para treinar o autoencoder, utilizou-se como critério e método de perca o MSE, que é o erro quadrado médio entre o que entrou no encoder e o que saiu no decoder. O MSE foi usado pois ele deixa os erros mais significativos com maior destaque, tendo em vista que eleva ao quadrado. Para otimizar, foi utilizado o método Adam para adaptar a taxa de aprendizagem, com learning rate (rt) = 0.001.

A equação do MSE pode ser vista a seguir:

![MSE](MSE.png)

# Etapa 2: Visualizando os embeddings

A mudança com o tunelamento mencionado na etapa anterior não trouxe resultados significantemente diferentes entre si, como pode ser visto na imagem a seguir:

![old_new](old_new.png)

Nota-se uma alteração no cluster 2, em que ele quase alcançava o -100 nas abscissas e ficava abaixo de 0 nas ordenadas. Porém, os embeddings na versão com tunelamento estão mais próximos do valor -80 nas abscissas e 0 nas ordenadas. Em relação ao cluster 1, ele estava entre 0 e -20 na versão original, e ficou entre -10 e 20 na nova. Ambos clusters 1 e 2 parecem ter se modificado, mas foi possível descobrir se o conteúdo/tema deles se alterou.

Ademais, é visualmente perceptível que não ocorreu uma mudança significativa na identificação dos demais clusters entre as duas versões.

# Etapa 3: Testando o sistema de pesquisa

