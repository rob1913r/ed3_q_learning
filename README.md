# ed3_q_learning

Q-learning tabular em uma grade 4x4, feito para o Exercício 6 do Estudo Dirigido 3 de
Fundamentos de Inteligência Artificial (EEE048, UFMG). O enunciado não faz parte do repositório.

## Problema

Um rover sai de uma célula e deve chegar à saída em (4, 4) com o menor número de passos,
sem atolar. As células são pares (coluna, linha), com a linha 1 na base.

| | coluna 1 | coluna 2 | coluna 3 | coluna 4 |
|---|---|---|---|---|
| **linha 4** | vazia | radiação | vazia | saída |
| **linha 3** | vazia | atoleiro | vazia | vazia |
| **linha 2** | vazia | vazia | vazia | atoleiro |
| **linha 1** | vazia | vazia | radiação | vazia |

A recompensa é a da célula de destino: vazia -1, atoleiro -5, radiação -20 e saída +20
(único estado terminal). Uma ação que sairia da grade não está disponível.

## Como rodar

```
python q_learning_grade.py
```

Precisa de Python 3.9 ou mais novo e não usa bibliotecas externas (testado em Python 3.14).

A saída mostra o trajeto e a recompensa total dos nove episódios (com cada atualização de Q
no primeiro), a tabela Q no início do 9º episódio e a probabilidade de escolher a ação ↓ na
célula (4, 3) sob ε-greedy.

## Parâmetros

α = 0,2, γ = 0,95 e ε = 0,1. Nos episódios a ação gananciosa é sempre a escolhida, com
desempate na ordem ↑ → ↓ ←, e Q começa em zero. Os estados iniciais dos nove episódios são
(2,2), (3,1), (1,1), (1,2), (1,4), (1,4), (2,2), (3,1) e (1,1). A atualização é

```
Q(s, a) <- (1 - α) Q(s, a) + α [ r + γ max Q(s', a') ]
```

com o máximo valendo 0 na célula terminal, que não tem ações.

## Estrutura

Tudo está em `q_learning_grade.py`:

- `Action`: as quatro ações, declaradas na ordem de desempate.
- `GridWorld`: a grade, as recompensas e as ações disponíveis em cada célula.
- `QLearning`: a tabela Q, a ação gananciosa, a atualização e a probabilidade sob ε-greedy.
- `Episode` e `Step`: o registro de cada episódio e das atualizações feitas.
