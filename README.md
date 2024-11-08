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

Resultado para a query "html":

```
{
  "results": [
    {
      "title": "Desenvolvedor Cobol Sênior",
      "content": "o mundo é ágil e nos cobra agilidade! acreditamos que a inovação é, antes de tudo, uma atitude. decidir, surpreender, testar e implantar novas realidades e redefinir o futuro. queremos criar ideias que transformem pessoas e, assim, impulsionar as pessoas a transformar o mundo. como conquistamos esta transformação: provocando a mudança da mentalidade de nossas pessoas - buscando empreendedorismo, resiliência, compreensão da proposta de valor qintess e vivência no dia a dia. sua missão estamos à p",
      "relevance": 0.21543465554714203
    },
    {
      "title": "Desenvolvedor Java/Spring Boot - Pleno/Sênior",
      "content": "responsabilidades: requisitos:",
      "relevance": 0.21543465554714203
    },
    {
      "title": "P&d | desenvolvedor (a) p&d -inteligência artificial e visão computacional",
      "content": "somos especializados no desenvolvimento e fabricação de rpa (remotely piloted aircrafts), sendo a maior empresa brasileira e latino-americana do segmento. fazer parte do surgimento e da consolidação de uma nova tecnologia, contribuir para seu aprimoramento e impactar vidas nos enche de orgulho e de vontade de querer fazer sempre mais. não é ficção, realmente fazemos robótica! atualmente, a companhia conta com mais de 500 colaboradores, xmobotianos (as) que visam tornar a robótica móvel uma reali",
      "relevance": 0.20180587470531464
    },
    {
      "title": "Desenvolvedor HCL Full Stack com Especialização em Site Builder",
      "content": "somos a maior multinacional brasileira e estamos entre as 100 maiores empresas de ti do mundo. estamos presentes em 41 países com +30k colaboradores. a stefanini tem como principal objetivo auxiliar os clientes a encontrar, por meio de soluções personalizadas, a maneira ideal para alcançar seus desafios, impulsionando a inovação digital. e queremos mais! buscamos profissional versátil e experiente em desenvolvimento de software, com especialização na plataforma hcl e profundo conhecimento tanto ",
      "relevance": 0.16796767711639404
    },
    {
      "title": "Desenvolvedor Full Stack",
      "content": "diferenciais: formação: horário de trabalho: sálario: escolaridade mínima:",
      "relevance": 0.16678853332996368
    },
    {
      "title": "Specialist Software Engineer",
      "content": "garantir a qualidade das entregas através de testes automatizados e observabilidade através de instrumentação do código; desenvolver sistemas baseado nos princípios de engenharia da unico (soluções simples, rápidas, seguras, escaláveis, mensuráveis, resilientes e que serão lembradas); atuar ativamente em discussões e decisões do produto ajudando no direcionamento técnico; garantir a simplicidade, eficiência, manutenibilidade e reaproveitamento de código; propor melhorias, novas tecnologias e nov",
      "relevance": 0.1608075350522995
    },
    {
      "title": "Desenvolvedor Web Pleno",
      "content": "responsabilidades: requisitos:",
      "relevance": 0.12015296518802643
    },
    {
      "title": "2024-034 desenvolvedor jr.",
      "content": "requisitos: automotivado, apaixonado por tecnologia, interessado em evoluir como desenvolvedor, goste de trabalhar em grupo e conhecimentos de desenvolvimento de software, para servidores, nuvem e ou mobile. desejável: como pessoa desenvolvedora, você trabalhará no desenvolvimento de novas funcionalidades para o sistema de gerenciamento do fluxo de tráfego aéreo brasileiro. nesta função, você irá: conhecimentos em backend e frontend web, sistemas em nuvem, uso de containers. javascript / typescr",
      "relevance": 0.12015296518802643
    },
    {
      "title": "Desenvolvedor Cobol Sênior",
      "content": "o mundo é ágil e nos cobra agilidade! acreditamos que a inovação é, antes de tudo, uma atitude. decidir, surpreender, testar e implantar novas realidades e redefinir o futuro. queremos criar ideias que transformem pessoas e, assim, impulsionar as pessoas a transformar o mundo. como conquistamos esta transformação: provocando a mudança da mentalidade de nossas pessoas - buscando empreendedorismo, resiliência, compreensão da proposta de valor qintess e vivência no dia a dia. sua missão estamos à p",
      "relevance": 0.12015296518802643
    },
    {
      "title": "Dev Fullstack Pl - Projeto Internacional",
      "content": "o que vamos construir juntos? essa posição é para trabalhar em um projeto internacional. são equipes globais e estão em vários lugares no mundo, proporcionando uma boa conexão com diversas culturas e perfis. é uma boa oportunidade para atuar com a metodologia ágil, que faz parte do nosso dia a dia! trabalharemos juntos para construir melhores cenários e desenvolver boas soluções. nossa equipe está procurando um desenvolvedor fullstack para se juntar ao nosso grupo. nosso roteiro está cheio de op",
      "relevance": 0.12015296518802643
    }
  ],
  "message": "OK"
}
```
Como podemos ver na imagem a seguir:

![html](html.png)

Para a query "python java django", os resultados obtidos são os 4 seguintes, com um grau de similaridade maior do que 0.2 (ou 20%):

```
{
  "results": [
    {
      "title": "QA - Júnior",
      "content": "foco e objetivo: sistema customer relationship management experiência: júnior (clt) carga horária diária: 8 horas período de trabalho: 09:00 às 18:00 descrição da vaga: estamos procurando um qa júnior talentoso e detalhista para se juntar ao nosso time de desenvolvimento. se você tem um olhar aguçado para detalhes, é apaixonado por qualidade de software e deseja iniciar ou aprimorar sua carreira em garantia de qualidade, esta oportunidade é perfeita para você! como qa júnior, você trabalhará em ",
      "relevance": 0.2575746476650238
    },
    {
      "title": "Desenvolvedor ERP Júnior",
      "content": "distribuidora de produtos farmacêuticos admite desenvolvedor erp júnior em bauru (jardim contorno). apoiar no desenvolvimento de softwares de computador através da linguagem de programação sap, abap ou linguagem zim. apoiar na elaboração de códigos de programação, organizando sua escrita, análise e desempenho, proporcionando maior confiabilidade e produtividade do mesmo. interação com operações sistêmicas e usuários para entendimento de contexto de negócio. id: 29368066886 benefícios a combinar ",
      "relevance": 0.24151532351970673
    },
    {
      "title": "Desenvolvedor back end - pleno",
      "content": "estamos à procura de um analista desenvolvedor pleno com experiência robusta na plataforma .net para se juntar à nossa equipe! se você é um profissional apaixonado por tecnologia, com forte conhecimento em desenvolvimento web e análise de sistemas, e quer fazer parte de projetos desafiadores e inovadores, essa é a sua oportunidade. responsabilidades e atribuições requisitos e qualificações diferenciais: analista desenvolvedor pleno com experiência na plataforma .net exigido conhecimentos sólidos",
      "relevance": 0.22656527161598206
    },
    {
      "title": "CAS | Pessoa Especialista de Desenvolvimento de Software - Back-end para Time Cross",
      "content": "cas (centro administrativo sicredi) - software engineer specialist back-end - time cross seja para atuar em um time de centro (atuação cross), auxiliando os times na evolução técnica, desenvolvimento, treinamento, pesquisa de mercado, desenvolvimento, entre outras atividades. #li-jc1 #li-remote no centro administrativo sicredi (cas), para as posições nas áreas de negócio adotamos o formato de trabalho híbrido que se consolidou em 3 dias presenciais, na sede da empresa, localizada na av. assis br",
      "relevance": 0.2102096825838089
    }
  ],
  "message": "OK"
}
```

E para a query "Desenvolvedor de software com experiência em fintechs", também com grau de similaridade maior do que 20%:

```
{
  "results": [
    {
      "title": "P&d | desenvolvedor (a) p&d -inteligência artificial e visão computacional",
      "content": "somos especializados no desenvolvimento e fabricação de rpa (remotely piloted aircrafts), sendo a maior empresa brasileira e latino-americana do segmento. fazer parte do surgimento e da consolidação de uma nova tecnologia, contribuir para seu aprimoramento e impactar vidas nos enche de orgulho e de vontade de querer fazer sempre mais. não é ficção, realmente fazemos robótica! atualmente, a companhia conta com mais de 500 colaboradores, xmobotianos (as) que visam tornar a robótica móvel uma reali",
      "relevance": 0.2758162319660187
    },
    {
      "title": "Desenvolvedor Cobol Sênior",
      "content": "o mundo é ágil e nos cobra agilidade! acreditamos que a inovação é, antes de tudo, uma atitude. decidir, surpreender, testar e implantar novas realidades e redefinir o futuro. queremos criar ideias que transformem pessoas e, assim, impulsionar as pessoas a transformar o mundo. como conquistamos esta transformação: provocando a mudança da mentalidade de nossas pessoas - buscando empreendedorismo, resiliência, compreensão da proposta de valor qintess e vivência no dia a dia. sua missão estamos à p",
      "relevance": 0.2504330277442932
    },
    {
      "title": "Desenvolvedor Java/Spring Boot - Pleno/Sênior",
      "content": "responsabilidades: requisitos:",
      "relevance": 0.2504330277442932
    },
    {
      "title": "Desenvolvedor Cobol Sênior",
      "content": "o mundo é ágil e nos cobra agilidade! acreditamos que a inovação é, antes de tudo, uma atitude. decidir, surpreender, testar e implantar novas realidades e redefinir o futuro. queremos criar ideias que transformem pessoas e, assim, impulsionar as pessoas a transformar o mundo. como conquistamos esta transformação: provocando a mudança da mentalidade de nossas pessoas - buscando empreendedorismo, resiliência, compreensão da proposta de valor qintess e vivência no dia a dia. sua missão estamos à p",
      "relevance": 0.21011720597743988
    },
    {
      "title": "2024-034 desenvolvedor jr.",
      "content": "requisitos: automotivado, apaixonado por tecnologia, interessado em evoluir como desenvolvedor, goste de trabalhar em grupo e conhecimentos de desenvolvimento de software, para servidores, nuvem e ou mobile. desejável: como pessoa desenvolvedora, você trabalhará no desenvolvimento de novas funcionalidades para o sistema de gerenciamento do fluxo de tráfego aéreo brasileiro. nesta função, você irá: conhecimentos em backend e frontend web, sistemas em nuvem, uso de containers. javascript / typescr",
      "relevance": 0.21011720597743988
    },
    {
      "title": "Desenvolvedor(a) Flutter - Vaga",
      "content": "empresa do segmento de tecnologia contrata desenvolvedor flutter sênior para atuar em projetos desafiadores de grandes clientes. requisitos escolaridade mínima jornada de trabalho benefícios experiência com flutter experiência com projetos de alta complexidade conhecimento de arquitetura de software experiência com métodos ágeis experiência com java assistência médica assistência odontológica seguro de vida gympass vale alimentação vale refeição café da manhã vale transporte bicicletário vestiár",
      "relevance": 0.21011720597743988
    },
    {
      "title": "Dev Fullstack Pl - Projeto Internacional",
      "content": "o que vamos construir juntos? essa posição é para trabalhar em um projeto internacional. são equipes globais e estão em vários lugares no mundo, proporcionando uma boa conexão com diversas culturas e perfis. é uma boa oportunidade para atuar com a metodologia ágil, que faz parte do nosso dia a dia! trabalharemos juntos para construir melhores cenários e desenvolver boas soluções. nossa equipe está procurando um desenvolvedor fullstack para se juntar ao nosso grupo. nosso roteiro está cheio de op",
      "relevance": 0.21011720597743988
    },
    {
      "title": "Desenvolvedor Web Pleno",
      "content": "responsabilidades: requisitos:",
      "relevance": 0.21011720597743988
    },
    {
      "title": "Desenvolvedor Backend Node JS + Java Sr (3627)",
      "content": "hello dev, você que deseja trabalhar em uma grande empresa, temos uma oportunidade para você. essa vaga é para atuar em hospital renomado de são paulo, com alta estrutura tecnológica, com equipes especializadas e que investe em ensino e pesquisa. não perca tempo e se cadastre, é rápido e fácil formação acadêmica modelo de contratação forma de atuação somos uma empresa de consultoria em ti com mais de 10 anos no mercado e contamos com um time de especialistas em recrutamento tech. nosso processo ",
      "relevance": 0.20137813687324524
    }
  ],
  "message": "OK"
}
```

O resultado inicial não é óbvio pois seu conteúdo não contém a palavra "fintech", retornando uma empresa que usa tecnologia para drones, veículos aereos e robótica geral. Porém, a empresa não é diretamente relacionada ao setor financeiro. 