# Projeto 3: Sistema de Análise de Texto com NLP

## Objetivo
Desenvolver um software para análise de texto que:
- Corrige erros ortográficos/semânticos.
- Classifica polaridade, subjetividade e sentimentos em frases.

## Requisitos Funcionais
### Input
- Frases em texto livre (ex: "Não sei quando vou viajar infelizmente").

### Output
1. **Frase Corrigida**  
   Ex: "Não sei quando vou viajar, infelizmente." (adiciona vírgula).
2. **Parâmetros Identificados**  
   - Tipo: `negação`  
   - Contexto: `pessoal`  
   - Sentimento: `tristeza`.

## Requisitos Técnicos
### Tecnologias
- **Linguagem**: Python (recomendado) ou outra de preferência.
- **Bibliotecas**:
  - Correção: `Hunspell`, `TextBlob`  
  - Análise Semântica: `spaCy`, `NLTK`  
  - Sentimentos: `VaderSentiment`, `Transformers`.

### Algoritmos
- **Correção**: Levenshtein Distance para sugestões de palavras.
- **Classificação**: Modelos pré-treinados (ex: `BERT` para subjetividade).

## Entregáveis
### Código
- Script principal (ex: `text_analyzer.py`).
- Requirements file (ex: `requirements.txt`).

### Apresentação
- **Duração**: 20 minutos.
- **Conteúdo**:
  ```markdown
  1. Demonstração ao vivo (5 min)
  2. Explicação das bibliotecas (7 min)
  3. Arquitetura do código (8 min)
  ```

### Relatório
- **Formato**: PDF, fonte Arial 11.
- **Estrutura**:
  ```markdown
  1. Introdução (objetivos)
  2. Metodologia (fluxograma do sistema)
  3. Exemplos de Input/Output (imagens)
  4. Dificuldades Técnicas
  ```

## Timeline
| Data       | Tarefa                             |
|------------|------------------------------------|
| 08/03      | Entrega do código (GitHub)         |
| 11/03      | Apresentação (20 min)              |
| 15/03      | Relatório final em PDF             |

## Notas
- **Testes**: Validar com frases ambíguas (ex: "Este filme não é ruim" → polaridade positiva).
- **Penalizações**: -10% por dia de atraso.
