# Projeto 3 & Final: Análise de Texto e Reconhecimento de Emoções em Voz

## Índice
- [Projeto 3 \& Final: Análise de Texto e Reconhecimento de Emoções em Voz](#projeto-3--final-análise-de-texto-e-reconhecimento-de-emoções-em-voz)
  - [Índice](#índice)
  - [Objetivo Geral](#objetivo-geral)
  - [Projeto 3: Análise de Texto](#projeto-3-análise-de-texto)
    - [Objetivos](#objetivos)
    - [Requisitos Funcionais](#requisitos-funcionais)
    - [Requisitos Técnicos](#requisitos-técnicos)
    - [Entregáveis](#entregáveis)
  - [Projeto Final: Reconhecimento de Voz](#projeto-final-reconhecimento-de-voz)
    - [Objetivos](#objetivos-1)
    - [Requisitos Funcionais](#requisitos-funcionais-1)
    - [Requisitos Técnicos](#requisitos-técnicos-1)
    - [Entregáveis](#entregáveis-1)
  - [Timeline](#timeline)
  - [Regras de Grupo](#regras-de-grupo)
  - [Apresentações](#apresentações)
  - [Relatório](#relatório)
  - [Penalizações](#penalizações)
  - [Notas Técnicas](#notas-técnicas)

---

## Objetivo Geral
Desenvolver um sistema multimodal que:
- Analisa texto (Projeto 3): Corrige erros e identifica parâmetros emocionais/semânticos.
- Analisa voz (Projeto Final): Reconhece emoções e dialetos a partir de input de áudio.

---

## Projeto 3: Análise de Texto
### Objetivos
- ✅ Corrigir frases com erros ortográficos/semânticos.
- ✅ Classificar sentimentos (ex: "alegria", "tristeza").
- ✅ Identificar polaridade (positivo/negativo) e subjetividade (factual/pessoal).

### Requisitos Funcionais
- Input: Frases em texto livre (ex: "Felizmente não chumbaram todos").
- Output:
  - Frase corrigida (ex: "Felizmente não chumbaram todos" → sem alteração).
  - Parâmetros:
    - **Tipo**: Afirmação/Negação  
    - **Contexto**: Factual/Pessoal  
    - **Sentimento**: Neutro/Alegria/Tristeza (ou outros).

### Requisitos Técnicos
- Linguagem: Qualquer (Python recomendado para NLP).
- Bibliotecas Sugeridas:
  - `spaCy` (análise gramatical)  
  - `TextBlob` (polaridade/subjetividade)  
  - `HunSpell` (correção ortográfica).

### Entregáveis
- Código funcional (arquivo `.zip` ou repositório GitHub).
- Apresentação (20 min):
  - Demonstração ao vivo.
  - Explicação das bibliotecas usadas.
- Relatório em PDF:
  - Capa, conteúdo, imagens/gráficos.
  - Formato: Fonte Arial 11, espaçamento 1.5.

---

## Projeto Final: Reconhecimento de Voz
### Objetivos
- 🎤 Adicionar input por voz ao Projeto 3.
- 🎯 Reconhecer dialetos (ex: diferenciar "v" e "b").
- 😊 Identificar emoções (ex: alegria, raiva) baseado em características vocais.

### Requisitos Funcionais
- Frases-chave pré-definidas (mínimo 4):
  - Ex: "Estou muito feliz com este resultado!"  
  - Ex: "Esta situação está-me a deixar frustrado."
- Output:
  - Transcrição corrigida.
  - Emoção detectada (ex: "Alegria", "Frustração").

### Requisitos Técnicos
- Reconhecimento de Voz:
  - `SpeechRecognition` (Python)  
  - `Web Speech API` (JavaScript).
- Análise de Emoções:
  - Extração de MFCCs (com `librosa`).  
  - Modelos pré-treinados (ex: `pyAudioAnalysis`).

### Entregáveis
- Código atualizado (integrar voz + texto).
- Relatório Final:
  - Comparação entre abordagens texto/voz.
  - Limitações encontradas.
- Apresentação (30 min):
  - Demo com microfone.
  - Análise de desempenho.

---

## Timeline
| Data          | Atividade                          |
|---------------|------------------------------------|
| 08/03         | Entrega código Projeto 3 (23h59)   |
| 11/03         | Apresentação Projeto 3 (20 min)    |
| 15/03         | Relatório Projeto 3 (23h59)        |
| 17/03         | Apresentação Projeto Final (30 min)|

---

## Regras de Grupo
- 👥 **Membros**: 3 elementos.
- 📁 **Nomenclatura de Arquivos**:
  - `Projeto_Final_NR_GRUPO.pdf` (ex: `Projeto_Final_05_GRUPO3.pdf`).

---

## Apresentações
| Projeto   | Duração | Conteúdo Obrigatório                          |
|-----------|---------|-----------------------------------------------|
| 3         | 20 min  | - Demo do software<br>- Explicação do código  |
| Final     | 30 min  | - Comparação texto/voz<br>- Análise de erros  |

---

## Relatório
- **Estrutura Mínima**:
  ```markdown
  1. Introdução
  2. Metodologia (bibliotecas/algoritmos)
  3. Implementação (fluxo do código)
  4. Resultados (exemplos + imagens)
  5. Conclusão (dificuldades + melhorias)
  ```

---

## Penalizações
- ⏰ **Atrasos**: -10% por dia.
- 🚫 **Plágio**: Nota zero.

---

## Notas Técnicas
- **Dialetos**: Usar datasets regionais (ex: [Common Voice](https://commonvoice.mozilla.org/)).
- **Emoções**: Priorizar 4 emoções básicas (alegria, tristeza, raiva, neutro).

---

❓ **FAQ**  
- *"Posso usar ChatGPT para correção de texto?"* → Sim, mas documentar no relatório.  
- *"Como lidar com dialetos?"* → Usar modelos treinados em datasets específicos (ex: `transformers` para fine-tuning). 
