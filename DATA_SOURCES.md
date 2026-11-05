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
| [GPT4-500k-Augmented-PTBR-Clean](https://huggingface.co/datasets/cnmoro/GPT4-500k-Augmented-PTBR-Clean) | MIT declarado na página | 565.536 registros; instruções e respostas traduzidas para português | Somente perfilamento exploratório enquanto a origem/licença é investigada | **Bloqueado para treinamento** até revisão específica dos outputs GPT-4 e da cadeia de direitos |
| [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | Apache-2.0 declarado na página | 161.443 mensagens no corpus completo; o card lista 2.969 em português (Brasil); o split HF ready-for-export tem 88.838 mensagens no total | Primeira fonte para perfilamento de mensagens pt-BR, qualidade, papéis e árvores de conversa | **Candidato prioritário para inspeção**; filtrar lang=pt e revisar synthetic/review_result/deleted |
| [MASSIVE pt-BR](https://huggingface.co/datasets/Magurofg/massive-pt-br) | CC BY 4.0 | Intenções e slots; a página informa 15.807 linhas no viewer | Apoiar classificação de intenção e extração/preservação de entidades e requisitos | **Candidato auxiliar**: atribuição obrigatória; não é dataset de geração R.C.F.O. |
| [IFEval-pt](https://huggingface.co/datasets/Magurofg/ifeval-pt) | Apache-2.0 | 308 itens pt-BR; inclui restrições próprias da língua portuguesa verificadas por código | Benchmark auxiliar de seguimento de instruções | **Candidato para avaliação**, não fonte de treinamento |
| [IFEval-PT](https://huggingface.co/datasets/Polygl0t/IFEval-PT) | Apache-2.0 | 300 prompts traduzidos para português | Benchmark traduzido complementar, após revisão linguística | **Candidato para avaliação**, não fonte de treinamento |

### 1. GPT4-500k-Augmented-PTBR-Clean

A página declara licença MIT, 565.536 linhas e descreve o conjunto como uma versão traduzida de Open-Orca/1million-gpt-4. Também informa que tarefas de programação e exemplos com caracteres não latinos foram removidos. A página informa que o campo response pode ter comprimento mínimo zero; portanto, respostas vazias precisam ser contadas e tratadas.

**Uso pretendido:** selecionar instruções e respostas úteis como matéria-prima para criar exemplos específicos do InspectorAI. Não copiar cegamente o campo de resposta como alvo R.C.F.O. ou CLARIFY.

**Verificações bloqueantes:**

- Open-Orca descreve o conjunto de origem como instruções FLAN aumentadas com completions GPT-4/GPT-3.5. Como o InspectorAI é um modelo próprio, há uma questão contratual relevante: os Termos de Uso da OpenAI em vigor na data desta revisão restringem o uso de outputs para desenvolver modelos que concorram com os da OpenAI. A licença MIT declarada pelo publicador não resolve, por si só, essa questão. Não usar respostas GPT-4 como alvos, nem selecionar/transformar essas respostas para treino, até obter uma revisão jurídica/contratual adequada. Esta pendência não impede apenas o perfilamento técnico da estrutura para fins exploratórios.

- Salvar a versão/commit, o README e a licença do dataset derivado.
- Examinar a licença, a descrição e a proveniência do Open-Orca/1million-gpt-4, inclusive os subdatasets FLAN mencionados pelo projeto Open-Orca.
- Investigar se a transformação/tradução preserva as condições aplicáveis às fontes anteriores.
- Inspecionar amostras de português brasileiro, traduções defeituosas, respostas vazias, conteúdo duplicado e dados sensíveis.
- Não liberar para uso comercial até documentar a análise da cadeia de direitos. A etiqueta MIT da página é uma declaração do publicador, não uma auditoria independente de todos os componentes de origem.

### 2. OpenAssistant OASST1

A página do dataset declara Apache-2.0 e descreve o corpus completo com 161.443 mensagens em 35 idiomas; a distribuição publicada lista 2.969 mensagens em português (Brasil). Atenção: a tabela Parquet usada por load_dataset expõe o subconjunto ready_for_export, com 88.838 mensagens divididas em train (84.437) e validation (4.401). A distribuição de idiomas precisa ser contada novamente nesses splits; não presumir que as 2.969 mensagens estejam todas neles. O esquema inclui role, lang, review_result, deleted, synthetic, parent_id e message_tree_id.

**Uso pretendido:** obter exemplos reais de formulação de pedidos e respostas em conversação. Usar somente mensagens em português brasileiro confirmadas pelos metadados; não inferir locale a partir de palavras isoladas.

**Verificações obrigatórias:**

- Contar todos os valores do campo lang nos splits train e validation antes de filtrar; não assumir que o total de 2.969 do corpus completo esteja presente no subconjunto ready_for_export.
- Contar mensagens e árvores utilizáveis em pt-BR; não confundir número de mensagens com número de pares de conversa.
- Separar role=prompter e role=assistant; contar synthetic, review_result e deleted antes de definir elegibilidade.
- Diferenciar respostas alternativas, mensagens sinalizadas e registros sem pai/contexto.
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

### 4. IFEval-pt (pt-BR)

A página declara Apache-2.0 e descreve 308 itens em português brasileiro, incluindo restrições próprias do idioma, como crase, próclise, ênclise, mesóclise, uso de cedilha e distinção entre por que/porque.

**Uso pretendido:** benchmark auxiliar de seguimento de instruções, com verificações por código. Não é um conjunto de exemplos de classificação de ambiguidade e não deve ser misturado automaticamente ao treinamento.

**Verificações obrigatórias:**

- Fixar a revisão do dataset usada.
- Preservar instruction_id_list e kwargs para que as verificações sejam reproduzíveis.
- Observar limitações estatísticas por restrição: há poucos itens por tipo de regra.
- Manter esse conjunto fora do treinamento se estiver sendo usado como teste.

### 5. IFEval-PT (Polygl0t, versão traduzida)

A página declara Apache-2.0 e contém 300 prompts traduzidos do benchmark original para português. É uma opção complementar, mas não representa necessariamente restrições específicas do português brasileiro.

**Uso pretendido:** benchmark secundário de instruções traduzidas, após verificar naturalidade e adequação ao pt-BR. Manter separado do IFEval-pt, pois os conjuntos medem aspectos diferentes.

## Fontes e referências

- [GPT4-500k-Augmented-PTBR-Clean — página e dataset card](https://huggingface.co/datasets/cnmoro/GPT4-500k-Augmented-PTBR-Clean)
- [Open-Orca/1million-gpt-4 — dataset de origem declarado](https://huggingface.co/datasets/Open-Orca/1million-gpt-4)
- [Open-Orca/OpenOrca — descrição das fontes FLAN](https://huggingface.co/datasets/Open-Orca/OpenOrca)
- [OpenAssistant OASST1 — página, licença e arquivos](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [OpenAssistant — FAQ sobre licença dos dados](https://projects.laion.ai/Open-Assistant/docs/faq)
- [MASSIVE pt-BR — página e dataset card](https://huggingface.co/datasets/Magurofg/massive-pt-br)
- [MASSIVE original — Amazon Science](https://huggingface.co/datasets/AmazonScience/massive)
- [IFEval-pt pt-BR — página e dataset card](https://huggingface.co/datasets/Magurofg/ifeval-pt)
- [IFEval-PT traduzido — página e dataset card](https://huggingface.co/datasets/Polygl0t/IFEval-PT)
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
