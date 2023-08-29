## Overfitting
<div align="left"> 
  <img align="center" src="https://img.shields.io/badge/Python-FF8C00?style=for-the-badge&logo=python&logoColor=white"><br>
  <img src="https://img.shields.io/badge/Ciência%20de%20Dados-red">  
  <img src="https://img.shields.io/badge/Aprendizado%20de%20Máquina-blue">
</div>

Prática de Ganho de Informação no contexto da disciplina Aprendizado de Máquina no CEFET-MG.

## Overfitting - Exemplo Ilustrativo
A principio, vamos fazer um dataset artificial em que possuimos dois atributos (também chamado de caracteríscas ou, do inglês, *features*) e duas possíveis saídas (também chamado de valor alvo ou classe alvo). Para isso, temos a matriz `x` e o vetor `y` em que, para cada exemplo `i`, cada linha `x[i]` dessa matriz representa esse exemplo (neste caso, representado por dois atributos) e a classe alvo `y[i]`.

Nessa e nas demais práticas, iremos representar a classe alvo como um vetor `y` e, os atributos, pela matriz `x`.

Abaixo, podemos ver a representação gráfica deste dataset em que, para cada instancia `i`, o eixo x é o atributo `x[i][0]` e o eixo y é o atributo `x[i][1]` a classe alvo `y[i]` é representada pela cor.

![image](https://github.com/aaugustoag/Overfitting/assets/49174397/8190329e-2451-453c-9f7c-4c1989363610)

##

Inicialmente, no aquivo `arvore_de_decisao.py` implemente a função `cria_modelo`.

Nessa função, você deverá criar um modelo baseado em arvore de decisão, por meio de um treino. Para o treino, use a variável `x` (que pode ser uma matriz ou DataFrame) em que, cada linha, é uma instância representada pelos seus atributos, além disso, a `y` é um vetor ou Series  represntando a classe alvo  de cada instância. Coloque como `random_state=1` que é o seed (semente) da função aleatória usada, pois, por padrão, a árvore de decisão do Scikit learn obtém os dados de forma aleátoria. Definindo este parametro, garantimos que o resultado será o mesmo em todas as execuções.

Além disso, com o objetivo de avaliarmos o overfitting, essa função possuirá o parametro `min_samples` que 
 que define o mínimo de exemplos necessários para que um nodo da árvore efetue a divisão. Use esse parâmetro ao instanciar a Árvore de Decisão. 

Para implementar essa função, use a classe `DecisionTreeClassifier`. [Veja a documentação desta classe](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html). Se necessário, comente a importação abaixo, copie e cole a função aqui e, logo após, volte ela para o arquivo. Para criar/obter o modelo use o [método fit](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.fit).

Abaixo, crie um modelo e use a função `plot_decision_boudary` para gerar o grafico apresentado o dataset ilustrativo com a superfície de decisão do modelo criado. Essa função está no arquivo `util.py`.

Na criação do modelo, altere o parametro `min_samples` até um valor que você julgue adequado. Veja o *overfitting* em valores muito baixos (abaixo de 1%, principalmente) e *underfitting* em valores muito altos.

![image](https://github.com/aaugustoag/Overfitting/assets/49174397/3f34e160-52b2-48d8-9a06-61bed34c14c0)

## Impacto do Overfitting/Underfitting - Estimativa Automática da Qualidade de Conteúdo
Nesta prática, iremos usar dados de 3.294 artigos da Wikipédia rotulados manualmente quanto a sua qualidade. 

Esses artigos passaram por uma avaliação pela comunidade de editores da Wikipedia. Tais editores classificaram esses artigos quanto a qualidade da seguinte forma: 

- **Artigo Destaque (FA)**: Os artigos atribuídos a esta classe são, de acordo com os avaliadores, os melhores artigos da Wikipédia.
- **Classe A (AC)**: os artigos da Classe A são considerados completos, mas com alguns problemas pendentes que precisam ser resolvidos para serem promovidos a Artigos em destaque.
- **Artigo Bons (GA)**: Bons Artigos são aqueles sem problemas de lacunas ou conteúdo excessivo. Essas são boas fontes de informação, embora outras enciclopédias possam fornecer um conteúdo melhor.
- **Classe B (BC)**: os artigos atribuídos a essa classe são considerados úteis para a maioria dos usuários, mas carecem de informações mais precisas.
- **Classe Inicial (ST)**: os artigos da Classe Inicial ainda estão incompletos, embora contenham referências e ponteiros para informações mais completas.
- **Artigos Rascunhos (SB)**: os artigos de toco são artigos de rascunho, com poucos parágrafos. Eles também têm poucas ou nenhumas citações.

Assim, [Dalip et. al. (2009)](https://dl.acm.org/citation.cfm?id=1555449) fizeram o preprocessamento desses artigos para serem extraídos indicadores de qualidades tais como: idade do artigo, tamanho, número de citações. Com tais indicadores e a classe de qualidade, foi possível realizar a predição automática de qualidade de artigos da Wikipédia.

Nesta prática, iremos fazer a previsão da qualidade usando os indicadores proposto por [Dalip et. al. (2009)](https://dl.acm.org/citation.cfm?id=1555449) e uma árvore de decisão.

Inicialmente, uso um DataFrame pandas e [leia o arquivo `wikipedia.csv`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) e exiba os dados deste dataset. Coloque como o rótulo da linha o id do artigo (ou seja, no dataset, a coluna `id` será a `index_col` do DataFrame).

Antes de executar a classificação e verificar o acerto no treino e teste, você deverá implementar a função `divide_treino_teste` que está no arquivo `arvore_decisao.py`.

Essa função deverá dividir os dados, de forma aleatoria, em treino e teste.  Para isso, faça o seguinte: 

1. Crie o DataFrame `df_treino` por meio do Dataframe `df` e a proporção `val_proporcao_treino`, passados como parâmetro. `val_proporcao_treino` assume um valor de 0 a 1 em que, por exemplo, 0.8 representa que 80% das instancias serão de treino e, o restante, o teste. . O [método `sample`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html) irá auxiliar para isso. Us, como parâmetro `random_state=1`. Esse será o valor da semente (seed) da função aleatória para manter sempre os dados embaralhados da mesma forma (o teste unitário só irá funcionar caso tenha colocado a esse valor de semente);

2. Conforme dito, o restante das instancias estarão em `df_teste`. Uma forma fácil de criar o `df_teste` é obter os elementos que estão em `df` e não estão em `df_treino`. Para isso, use [o método `drop`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html) que elimina conlunas ou  linhas de um DataFrame. Para eliminar as linhas, obtenha o id de cada linha do treino usando `df_treino.index`.

A seguir, execute a função `divide_treino_teste` com uma divisão de 80% de treino e, logo após, usando df_treino e df_teste, crie as seguintes variáveis:
-  `x_treino` : DataFrame que representa, para cada linha do **treino**, todos os atributos de um exemplo do treino. Para isso, elimine a coluna que representa a classe por meio [método `drop` do DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html), alterando o parametro axis;
- `y_treino`: Series que representa, para cada posição `i`, a classe alvo do exemplo `i` representado pelos atributos `x_treino[i]`. A classe alvo está na coluna `realClass`;
- `x_teste`: Similar ao `x_teste`, porém com as instancias do **teste**; 
- `y_teste`: Similar ao `y_treino`, porém, são as classe alvo do teste.

Implemente agora o método `faz_classificacao`. Ele passará como parametro as variáveis `X_treino`, `y_treino`, `X_teste`, `y_teste`, criadas anteriormente além do parâmetro `min_samples` que define a quantidade mínima de instancias para que se divida um nodo da árvore de decisão.

Assim, esta função irá:

1- Criar o modelo a partir dos dados de treino e o parametro `min_samples` (você pode usar a função criada anteriormente);

2- Realizar a predição usando o [método predict](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.predict). Esse método retorna uma lista de predição em que, para cada posição `i`, retorna o resultado previsto do exemplo representado por `x_teste[i]`;

3- A partir da lista obtida pela predição e da variável `y_teste`, calcule a `acuracia` que é a proporção de acertos, ou seja, $acuracia = acertos/|y_{teste}|$ em que `acertos` é a quantidade de acertos da predição e $y_{teste}$ é a lista `y_teste`.

![image](https://github.com/aaugustoag/Overfitting/assets/49174397/9d4f0c5d-c645-43b5-a1ac-1d0624636f06)

Por meio da função `plot_performance_min_samples` crie um gráfico em que o eixo `x` é a variação do parâmetro `min_samples` e, o eixo `y`, representará a acurácia. Você deverá veriar o `min_samples` de 0.001 até 0.7 de 0.01 em 0.01 passos. Esse gráfico possuirá duas linhas: representando a **acurácia no treino** durante a variação do `min_samples` e, a outra, a **acurácia do teste** com os diversos valores de `min_sample`.

- foi usada a função arange do numpy para o for (ao invés de range). Pois o range permite apenas passos com valores inteiros;
- para obter a acurácia no treino, o teste deverá possuir as mesmas instancias do treino;
- Para plotar foi usado o matplotlib veja: [https://matplotlib.org/users/pyplot_tutorial.html](https://matplotlib.org/users/pyplot_tutorial.html).

Execute abaixo a função plot_performance_min_samples usando as veriáveis X_treino,y_treino,X_teste,y_teste.<br>
![image](https://github.com/aaugustoag/Overfitting/assets/49174397/c2400bd9-2f60-48ee-8b7b-faf023c7f0f4)
