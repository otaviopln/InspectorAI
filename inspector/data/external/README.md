# Dados externos do InspectorAI

Este diretório é reservado para downloads e caches locais de datasets de terceiros usados na experiência pt-BR.

## Regras

- Não versionar datasets grandes ou arquivos binários de terceiros por padrão.
- Não modificar os arquivos originais baixados.
- Registrar dataset, URL, commit/revisão, licença, data de obtenção e SHA-256 em um manifesto.
- Guardar transformações em diretórios separados (interim/, annotated/ e processed/).
- Só utilizar uma fonte depois da revisão de licença, procedência, conteúdo e adequação.
- Respeitar avisos de atribuição e demais obrigações da licença.
- Não armazenar dados pessoais/sensíveis desnecessários.

As fontes candidatas e suas pendências estão documentadas em [DATA_SOURCES.md](../../../DATA_SOURCES.md). O plano de migração está em [docs/PTBR_MIGRATION_PLAN.md](../../../docs/PTBR_MIGRATION_PLAN.md).

Estrutura local sugerida:

    external/
    ├── README.md
    ├── raw/       # downloads originais; ignorados pelo Git
    └── cache/     # cache de processamento; ignorado pelo Git
