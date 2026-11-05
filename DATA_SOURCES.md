# Fontes de dados — InspectorAI (pt-BR)

> **Estado:** registro inicial de candidatos. Nenhum dataset externo foi importado ou aprovado para treinamento nesta etapa.
>
> **Última verificação das páginas:** 2026-11-05.
>
> Este arquivo registra metadados declarados pelos projetos de origem. Não é parecer jurídico e não substitui a leitura dos arquivos de licença, das condições de origem e da versão exata que será baixada.

## Política de admissão

Um dataset só poderá entrar no pipeline depois de:

1. Identificar a página oficial, autores/organização, versão ou commit e arquivos efetivamente utilizados.
2. Arquivar a licença aplicável e a atribuição exigida.
3. Investigar fontes anteriores quando o conjunto for derivado ou traduzido.
4. Confirmar que o uso comercial é permitido, sem cláusula NC ou outra restrição incompatível.
5. Verificar conteúdo, idioma, qualidade, duplicações, dados pessoais e adequação à tarefa.
6. Registrar transformações, filtros, quantidades e hashes.

**"Público", "open source", "open weights" e "disponível no Hugging Face" não são sinônimos de autorização irrestrita.** A licença de software do InspectorAI não substitui nem altera as licenças dos dados de terceiros.

## Registro de candidatos

| Fonte | Licença declarada na página consultada | Conteúdo / tamanho indicado | Uso proposto | Estado |
|---|---|---|---|---|
| [GPT4-500k-Augmented-PTBR-Clean](https://huggingface.co/datasets/cnmoro/GPT4-500k-Augmented-PTBR-Clean) | MIT | 565.536 registros; instruções e respostas traduzidas para português | Fonte inicial de solicitações e respostas para seleção, normalização e adaptação | **Condicional**: verificar licença e procedência de toda a cadeia de origem |
| [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | Apache-2.0 | Conversas humanas multilíngues, organizadas em mensagens/árvores | Fonte complementar para linguagem natural, pedidos e contexto conversacional; filtrar locale pt-BR | **Candidato**: medir volume pt-BR e conferir estrutura/licença no snapshot baixado |
| [MASSIVE pt-BR](https://huggingface.co/datasets/Magurofg/massive-pt-br) | CC BY 4.0 | Intenções e slots; a página informa 15.807 linhas no viewer | Apoiar classificação de intenção e extração/preservação de entidades e requisitos | **Candidato auxiliar**: atribuição obrigatória; não é dataset de geração R.C.F.O. |
| [IFEval-PT](https://huggingface.co/datasets/Polygl0t/IFEval-PT) | Apache-2.0 | 300 prompts com instruções verificáveis | Benchmark auxiliar de seguimento de instruções em português | **Candidato para avaliação**, não fonte principal de treinamento |

### 1. GPT4-500k-Augmented-PTBR-Clean

A página declara licença MIT, 565.536 linhas e descreve o conjunto como uma versão traduzida de Open-Orca/1million-gpt-4. Também informa que tarefas de programação e exemplos com caracteres não latinos foram removidos.

**Uso pretendido:** selecionar instruções e respostas úteis como matéria-prima para criar exemplos específicos do InspectorAI. Não copiar cegamente o campo de resposta como alvo R.C.F.O. ou CLARIFY.

**Verificações bloqueantes:**

- Salvar a versão/commit, o README e a licença do dataset derivado.
- Examinar a licença, a descrição e a proveniência do Open-Orca/1million-gpt-4, inclusive os subdatasets FLAN mencionados pelo projeto Open-Orca.
- Investigar se a transformação/tradução preserva as condições aplicáveis às fontes anteriores.
- Inspecionar amostras de português brasileiro, traduções defeituosas, respostas vazias, conteúdo duplicado e dados sensíveis.
- Não liberar para uso comercial até documentar a análise da cadeia de direitos. A etiqueta MIT da página é uma declaração do publicador, não uma auditoria independente de todos os componentes de origem.

### 2. OpenAssistant OASST1

A página do dataset declara Apache-2.0 e descreve um corpus multilíngue de conversas humanas.

**Uso pretendido:** obter exemplos reais de formulação de pedidos e respostas em conversação. Usar somente mensagens em português brasileiro confirmadas pelos metadados; não inferir locale a partir de palavras isoladas.

**Verificações obrigatórias:**

- Contar mensagens, conversas/árvores e pares utilizáveis em pt-BR no snapshot escolhido.
- Diferenciar mensagens de usuário e assistente, respostas alternativas, mensagens sinalizadas e registros sem pai/contexto.
- Preservar o identificador da conversa/árvore para impedir que mensagens correlatas apareçam em splits diferentes.
- Não converter uma resposta conversacional em alvo do InspectorAI sem adaptação e validação.

### 3. MASSIVE pt-BR

A página declara CC BY 4.0 e descreve uma localização brasileira do split pt-PT do MASSIVE, com rótulos de intenção e slots.

**Uso pretendido:** apoiar tarefas auxiliares de compreensão: reconhecer intenções, entidades e atributos explícitos. O dataset não deve ser tratado como se já contivesse pares de pergunta/resposta para geração CLARIFY ou R.C.F.O.

**Obrigações:**

- Atribuir o dataset derivado e citar também o trabalho original MASSIVE conforme indicado pelos autores.
- Registrar as alterações e a localização pt-BR realizadas.
- Manter os rótulos e a ligação entre frase e slots; não descartar os valores anotados.
- Verificar o repositório/procedimento de localização antes de assumir que todas as anotações foram validadas manualmente.

### 4. IFEval-PT

A página declara Apache-2.0 e apresenta 300 prompts traduzidos para português, com identificadores de instruções verificáveis e argumentos.

**Uso pretendido:** benchmark auxiliar para verificar cumprimento de restrições explícitas. Não é um conjunto de exemplos de classificação de ambiguidade e não deve ser misturado automaticamente ao treinamento.

**Verificações obrigatórias:**

- Confirmar se a variante linguística atende ao padrão pt-BR desejado.
- Preservar instruction_id_list e kwargs para que as verificações sejam reproduzíveis.
- Manter esse conjunto fora do treinamento se estiver sendo usado como teste.

## Fontes e referências

- [GPT4-500k-Augmented-PTBR-Clean — página e dataset card](https://huggingface.co/datasets/cnmoro/GPT4-500k-Augmented-PTBR-Clean)
- [Open-Orca/1million-gpt-4 — dataset de origem declarado](https://huggingface.co/datasets/Open-Orca/1million-gpt-4)
- [Open-Orca/OpenOrca — descrição das fontes FLAN](https://huggingface.co/datasets/Open-Orca/OpenOrca)
- [OpenAssistant OASST1 — página, licença e arquivos](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [OpenAssistant — FAQ sobre licença dos dados](https://projects.laion.ai/Open-Assistant/docs/faq)
- [MASSIVE pt-BR — página e dataset card](https://huggingface.co/datasets/Magurofg/massive-pt-br)
- [MASSIVE original — Amazon Science](https://huggingface.co/datasets/AmazonScience/massive)
- [IFEval-PT — página e dataset card](https://huggingface.co/datasets/Polygl0t/IFEval-PT)
- [Texto oficial MIT](https://opensource.org/license/mit)
- [Texto oficial Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [Texto oficial Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)

## Regra de decisão

Status possíveis:

- **candidate**: identificado, ainda não auditado integralmente.
- **conditional**: há pendência específica que bloqueia uso.
- **approved**: licença, origem, qualidade e uso pretendido foram revisados e documentados.
- **rejected**: licença, origem, privacidade ou qualidade incompatível.

**Nenhum candidato listado acima está automaticamente aprovado para treinamento ou distribuição comercial do InspectorAI.** A aprovação é por fonte, versão e uso pretendido.
