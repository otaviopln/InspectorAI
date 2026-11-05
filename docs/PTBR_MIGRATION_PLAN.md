# Plano de migração para português brasileiro — InspectorAI

**Branch de trabalho:** experiment/ptbr-dataset  
**Branch-base:** hardening/urgent-pipeline  
**Estado:** planejamento e inventário inicial; nenhuma migração de código ou importação de dados externos foi executada.  
**Idioma-alvo:** português brasileiro (pt-BR).

## 1. Decisão de projeto

Suspender a expansão manual do corpus em inglês e iniciar uma linha experimental isolada para investigar datasets públicos com licença compatível com uso comercial, procedência documentada e adequação ao objetivo do InspectorAI.

O objetivo do projeto permanece o mesmo:

1. Determinar se uma solicitação é suficientemente clara ou se precisa de esclarecimento.
2. Preservar o que o usuário já informou.
3. Identificar somente lacunas relevantes.
4. Reestruturar solicitações claras em Role, Context, Focus e Output (R.C.F.O.), sem acrescentar fatos, requisitos ou entregáveis não fundamentados.

A mudança é de idioma e estratégia de dados, não de objetivo funcional.

## 2. Princípios e limites

- O corpus inglês atual deve permanecer intacto como referência experimental.
- O novo trabalho deve permanecer nesta branch até que os testes e a avaliação justifiquem integração.
- Dados externos são matéria-prima; não são automaticamente exemplos corretos para CLARIFY ou R.C.F.O.
- Preservar arquivos originais e metadados. Transformações devem gerar artefatos derivados, nunca sobrescrever a origem.
- Cada registro derivado deve manter proveniência suficiente para rastrear dataset, versão, arquivo, ID original e transformações.
- Não usar datasets com cláusula de uso não comercial (NC), termos incompatíveis ou licença/proveniência não resolvida.
- Não presumir que a licença do repositório do InspectorAI se aplica aos dados externos.
- Não treinar com benchmark de teste.
- Separar avaliação estrutural, avaliação semântica e avaliação linguística.
- Não aumentar a arquitetura ou iniciar novo treinamento grande antes de validar os dados e o pipeline.

## 3. Diagnóstico que motiva a migração

O corpus inglês atualmente auditado contém 520 exemplos aprovados, todos com origem declarada assistant_seed, distribuídos em 416 treino, 52 validação e 52 teste. Todos os 520 registros possuem group_id único, ou seja, nenhum grupo reúne paráfrases ou variantes relacionadas.

A avaliação recente apresentou:

- acurácia de decisão: 90,91%;
- validade estrutural estrita: 43,64%;
- integridade dos campos R.C.F.O.: 42,11%;
- duplicação de perguntas em 41,67% dos casos CLARIFY;
- cobertura dos requisitos na suíte semântica anotada: 7,69% (13 requisitos, dos quais 1 preservado, em somente três casos).

Esses números não provam que o idioma inglês seja a causa. A migração para pt-BR é uma decisão estratégica e uma nova linha experimental; os problemas de fidelidade, estrutura e qualidade dos dados continuam precisando ser tratados explicitamente.

## 4. Fontes candidatas e papel de cada uma

O inventário atualizado e as observações de licença ficam em [DATA_SOURCES.md](../DATA_SOURCES.md).

| Dataset | Papel proposto | Não fazer |
|---|---|---|
| GPT4-500k-Augmented-PTBR-Clean | Apenas perfilamento técnico exploratório enquanto a procedência/licença é investigada | Não usar outputs GPT-4 como alvos ou sinais de treino; há restrições contratuais de uso de outputs para desenvolvimento de modelos concorrentes |
| OpenAssistant OASST1 | Primeira fonte para perfilamento e possível extração de conversas pt-BR | Filtrar lang=pt, inspecionar synthetic/review_result/deleted e preservar message_tree_id; não tratar toda mensagem como par independente |
| MASSIVE pt-BR | Sinais auxiliares de intenção e extração de slots/requisitos | Não tratá-lo como corpus de geração R.C.F.O. |
| IFEval-PT | Benchmark auxiliar de cumprimento de restrições | Não misturar o benchmark reservado para teste ao treinamento |

A primeira fonte a perfilar tecnicamente será OASST1. O card lista 2.969 mensagens brasileiras no corpus completo de 161.443 mensagens, mas o split Parquet ready_for_export usado pelo load_dataset contém 88.838 mensagens (84.437 train e 4.401 validation). A contagem de lang deve ser refeita em cada split; não assumir que as 2.969 estejam todas no subconjunto pronto para SFT. Mensagens também não equivalem a pares SFT. MASSIVE pt-BR deve ser avaliado como recurso auxiliar de compreensão/slots. IFEval-pt é principalmente avaliação. GPT4-500k pode ser inspecionado para entender schema e qualidade, mas fica bloqueado para treinamento até resolver a questão de uso dos outputs GPT-4 e a cadeia de direitos.

## 5. Fases de execução

### Fase 0 — Isolamento e preservação

- [x] Criar a branch experiment/ptbr-dataset a partir de hardening/urgent-pipeline.
- [x] Registrar a estratégia e fontes candidatas.
- [ ] Congelar os hashes do corpus, tokenizer e checkpoint ingleses como baseline.
- [ ] Confirmar que nenhum arquivo da linha experimental altera os artefatos da linha anterior.
- [ ] Definir os diretórios locais de download e cache; não versionar arquivos grandes de origem sem necessidade.

**Critério de saída:** baseline inglês preservado e branch experimental isolada.

### Fase 1 — Due diligence de licença e procedência

Para cada dataset/versão:

1. Baixar ou consultar o README, a licença integral, a configuração e os metadados da versão exata.
2. Registrar URL, commit/revisão, data de obtenção, autores, licença declarada e arquivos utilizados.
3. Em datasets derivados, identificar as fontes anteriores e analisar as condições herdadas.
4. Identificar exigências de atribuição, compartilhamento, aviso de alterações e eventuais restrições.
5. Verificar políticas de dados pessoais e conteúdo sensível.
6. Registrar decisão: approved, conditional ou rejected.

**Critério de saída:** nenhuma fonte entra no pipeline de treino sem uma decisão documentada compatível com o uso pretendido.

### Fase 2 — Inspeção e perfil dos dados brutos

Para cada fonte em inspeção:

- contar registros por idioma/locale e split;
- identificar esquema, campos e tipos;
- calcular proporção de campos ausentes/vazios;
- identificar duplicatas exatas e aproximadas;
- medir comprimentos de entrada e resposta;
- detectar mistura de idiomas e traduções pouco naturais;
- identificar respostas truncadas, vazias, genéricas ou inconsistentes;
- inspecionar amostras aleatórias e casos extremos;
- verificar conteúdo pessoal/sensível;
- avaliar se o registro é solicitação de usuário, resposta de assistente, rótulo de intenção ou outra modalidade.

Gerar um relatório por fonte antes de qualquer transformação. O script inicial está disponível em scripts/profile_external_dataset.py. Ele usa streaming, fixa o commit resolvido e salva apenas uma amostra limitada e um relatório no diretório ignorado inspector/data/external/cache/.

Exemplo para OASST1 (filtra o locale pt antes da amostragem):

    python -m pip install datasets huggingface_hub
    python scripts/profile_external_dataset.py --dataset OpenAssistant/oasst1 --split train --filter-field lang --filter-value pt --sample-size 200 --buffer-size 10000 --seed 42 --output-dir inspector/data/external/cache/oasst1_ptbr

Exemplo de perfil exploratório do GPT4-500k, sem permissão para treinamento:

    python scripts/profile_external_dataset.py --dataset cnmoro/GPT4-500k-Augmented-PTBR-Clean --split train --sample-size 200 --buffer-size 10000 --seed 42

A amostragem é exploratória e não estima com precisão a distribuição de todo o corpus. Para OASST1, o relatório precisa mostrar role, synthetic, review_result, deleted e message_tree_id; para GPT4-500k, deve verificar especialmente respostas vazias.

**Critério de saída:** conhecer o que cada arquivo contém e quais subconjuntos são utilizáveis.

### Fase 3 — Normalização e filtragem

Criar um pipeline reproduzível que:

- normalize espaços e quebras de linha sem reescrever o significado;
- padronize os rótulos de idioma, preservando locale original;
- remova duplicatas com critérios registrados;
- exclua registros incompletos ou sem relação com o objetivo;
- preserve IDs e metadados de origem;
- registre motivo e contagem de cada exclusão;
- não misture fontes sem manter a proveniência.

A normalização não deve traduzir ou parafrasear automaticamente os exemplos nesta fase.

**Critério de saída:** subconjuntos limpos, rastreáveis e reproduzíveis.

### Fase 4 — Transformação em tarefas InspectorAI

Os dados selecionados precisam ser adaptados às tarefas do projeto.

#### CLARIFY

A saída deve distinguir:

- **Requisitos conhecidos**: somente informações presentes no pedido;
- **Informações ausentes**: lacunas materiais para a execução;
- **Perguntas**: perguntas claras, relevantes, não duplicadas e que não solicitem novamente algo já informado.

Não inventar domínio, público, tecnologia, formato, orçamento, prazo ou contexto.

#### FORMAT_RCFO

A saída deve conter exatamente:

- **Decisão: FORMAT_RCFO**
- **Papel:**
- **Contexto:**
- **Foco:**
- **Saída:**

Cada campo precisa ser semanticamente fiel:

- **Papel:** função adequada à tarefa, sem criar autoridade ou especialização desnecessária.
- **Contexto:** circunstâncias fornecidas pelo usuário e informações estritamente necessárias.
- **Foco:** objetivo, operações, restrições, quantidades, datas e critérios explicitados.
- **Saída:** entregável/formato pedido, sem substituí-lo por outro.

Não converter a resposta original de um dataset de instruções em alvo R.C.F.O. por simples troca de cabeçalhos.

#### Proveniência de cada registro

O esquema derivado deve preservar, no mínimo:

    {
      "id": "ptbr_000001",
      "task": "clarify",
      "ambiguity": "partial",
      "language": "pt-BR",
      "input": "Solicitação original...",
      "target": "Saída adaptada para o InspectorAI...",
      "source_dataset": "identificador-do-dataset",
      "source_version": "commit-ou-versão",
      "source_record_id": "id-original",
      "source_license": "licença",
      "group_id": "familia-semantica",
      "transformation": "normalization-and-annotation-v1",
      "review_status": "pending_review"
    }

Os campos de origem devem ser preenchidos com metadados verificáveis; não inventar IDs ou versões. O esquema pode evoluir, mas a proveniência não pode ser descartada.

**Critério de saída:** pares input-target adequados ao InspectorAI, com fonte e transformação rastreáveis.

### Fase 5 — Agrupamento e splits sem vazamento

- Agrupar paráfrases, traduções, variantes de uma mesma intenção e mensagens da mesma conversa/árvore.
- Atribuir a família inteira a apenas um split.
- Separar treino, validação e teste antes da geração de variantes que poderiam vazar entre conjuntos.
- Manter manifesto de splits versionado e estável.
- Verificar sobreposição por texto normalizado e similaridade semântica.
- Garantir representação suficiente de vague, partial, clear e das duas decisões em cada split, quando os dados permitirem.

**Critério de saída:** relatório de vazamento semântico e manifesto reproduzível.

### Fase 6 — Benchmark pt-BR e avaliação

Criar um benchmark reservado, independente dos dados de treinamento, com casos de vários domínios e níveis de ambiguidade.

Medir separadamente:

- acurácia de decisão;
- validade estrutural;
- preservação de requisitos;
- taxa de conteúdo não fundamentado;
- relevância e não duplicação de perguntas;
- fidelidade por campo R.C.F.O.;
- qualidade linguística em pt-BR.

IFEval-PT pode complementar a avaliação de restrições verificáveis, mas não substitui o benchmark específico de CLARIFY/R.C.F.O.

A revisão humana pode ser feita por amostragem estratificada e revisão de todos os casos sinalizados pelo validador. As decisões e justificativas devem ser salvas.

**Critério de saída:** benchmark congelado, com regras, rótulos e critérios de aprovação documentados.

### Fase 7 — Adaptação técnica e treinamento experimental

Somente depois das fases anteriores:

1. Atualizar prepare_dataset.py para aceitar pt-BR sem enfraquecer validações.
2. Atualizar os cabeçalhos e o validador estrutural para português.
3. Adaptar tokenizer ao corpus de treino em pt-BR e medir cobertura/fragmentação.
4. Verificar comprimentos de entrada e saída; definir MAX_LEN com base em percentis observados, não por palpite.
5. Manter o modelo e hiperparâmetros constantes no primeiro experimento para isolar o efeito dos dados.
6. Testar se o modelo consegue memorizar um subconjunto pequeno e revisado.
7. Treinar apenas após passar nos testes de preparação, integridade e avaliação.
8. Comparar com a baseline inglesa usando métricas compatíveis e reportar limitações.

**Critério de saída:** resultados reproduzíveis, checkpoint/tokenizer/dataset pareados e nenhuma regressão ocultada.

## 6. Arquitetura de diretórios proposta

A estrutura pode ser implementada incrementalmente:

    inspector/data/
    ├── raw/                         # corpus antigo e fontes controladas do projeto
    ├── external/
    │   ├── README.md                # regras de armazenamento
    │   ├── raw/                     # downloads locais, não versionados
    │   └── cache/                   # cache local, não versionado
    ├── interim/                     # dados filtrados/normalizados
    ├── annotated/                   # dados transformados para InspectorAI
    └── processed/                   # splits e artefatos derivados

Não versionar indiscriminadamente datasets de centenas de MB. Versionar scripts, manifestos, documentação, relatórios resumidos, hashes e amostras pequenas permitidas. O download deve ser reproduzível a partir do identificador/commit documentado.

## 7. Critérios de aceitação para dados derivados

Um exemplo não será aprovado apenas por possuir JSON válido ou cabeçalhos corretos.

### Regras obrigatórias

- A decisão corresponde à ambiguidade definida.
- A resposta é derivada da solicitação, não de um template genérico.
- Todos os requisitos explícitos relevantes foram preservados.
- Nenhum requisito, contexto ou entregável sem fundamento foi adicionado.
- Perguntas CLARIFY investigam lacunas reais e não se repetem.
- Campos R.C.F.O. são completos, coerentes e semanticamente fiéis.
- Idioma e variante linguística foram identificados.
- Fonte, versão, ID de origem e transformação são rastreáveis.
- O registro está livre de duplicata indevida e conteúdo pessoal/sensível incompatível.

### Estados de revisão

- **pending_review**: candidato ainda não aprovado.
- **approved**: passou por todas as validações exigidas para seu uso.
- **rejected**: defeito irrecuperável, fonte incompatível ou conteúdo inadequado.
- **corrected**: registro foi corrigido; deve voltar à validação e revisão antes de ser aprovado.

Não marcar todo o dataset como approved por padrão. Aprovação precisa representar uma decisão real.

## 8. Regras para idioma

Idioma principal de produto: pt-BR.

- Respostas da aplicação devem ser em português brasileiro.
- Preservar nomes de bibliotecas, linguagens, APIs e termos técnicos quando forem próprios.
- Manter o idioma original do registro em metadados.
- Não misturar pt-PT com pt-BR sem marcar variante e executar revisão.
- Se forem aceitos outros idiomas no futuro, eles devem ter avaliação e configuração explícitas.

## 9. O que não fazer nesta branch ainda

- Não apagar nem reescrever os 520 exemplos ingleses.
- Não substituir o checkpoint oficial da linha anterior.
- Não treinar com o dataset público inteiro sem inspeção.
- Não usar IFEval-PT como material de treinamento quando estiver reservado para teste.
- Não declarar todos os datasets "MIT" ou "open source".
- Não misturar fontes sem preservar licença e proveniência por registro.
- Não alterar arquitetura, beam search ou hiperparâmetros para compensar dados ruins.
- Não anunciar suporte confiável a pt-BR antes de benchmark e revisão.

## 10. Próxima sequência imediata

1. Finalizar o inventário de licença/procedência de GPT4-500k-PTBR-Clean.
2. Baixar apenas metadados e uma amostra controlada das fontes aprovadas.
3. Criar um script de perfil de datasets que emita contagens, idioma, campos ausentes, duplicatas, comprimentos e amostras.
4. Inspecionar manualmente o resultado do perfil.
5. Selecionar a primeira fonte para transformação e preservar as demais como benchmark/auxiliares.
6. Só então implementar o normalizador/adaptador pt-BR.

## 11. Definição de pronto para a migração

A migração não será considerada pronta só porque o projeto aceita strings em português.

Ela estará pronta para treinamento quando houver:

- fontes e licenças documentadas;
- cadeia de origem revisada;
- dados rastreáveis;
- conjunto pt-BR limpo e adaptado;
- regras de anotação aplicadas;
- grupos e splits sem vazamento;
- benchmark independente;
- testes de pipeline aprovados;
- hashes dos dados, tokenizer e checkpoint registrados.

---

**Direção:** primeiro construir um pipeline de dados pt-BR confiável; depois decidir o treinamento. O valor central do InspectorAI continua sendo a análise de ambiguidade e a preservação semântica das solicitações.
