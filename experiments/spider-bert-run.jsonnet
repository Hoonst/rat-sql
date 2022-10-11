{
    logdir: "logdir/bert_run",
    model_config: "configs/spider/nl2code-bert.jsonnet",
    model_config_args: {
        data_path: 'data/spider/',
        bs: 6,
        num_batch_accumulated: 4,
        # plm_version: "google/electra-large-discriminator",
        # plm_version: "microsoft/deberta-v3-large",
        # plm_version: "microsoft/deberta-large",
        plm_version: "bert-large-uncased-whole-word-masking",
        summarize_header: "avg",
        use_column_type: false,
        max_steps: 120500,
        num_layers: 8,
        lr: 7.44e-4,
        plm_lr: 3e-6,
        att: 1,
        loss: "label_smooth",
        end_lr: 0,
        sc_link: true,
        cv_link: true,
        # dp_link: 'stanza', # if this is True, dist_relation should be False
        dp_link: 'spacy',
        qv_link: false,
        dist_relation: false,
        use_orthogonal: false,
        use_orth_init: false,
        bi_way: true,
        bi_match: true,
        use_align_mat: true,
        use_align_loss: true,
        plm_token_type: true,
        decoder_hidden_size: 512,
        end_with_from: true, # equivalent to "SWGOIF" if true
        clause_order: null, # strings like "SWGOIF", it will be prioriotized over end_with_from 
    },

    eval_name: "bert_run_%s_%d" % [self.eval_use_heuristic, self.eval_beam_size],
    eval_output: "__LOGDIR__/ie_dirs",
    eval_beam_size: 1,
    eval_use_heuristic: true,
    # eval_steps: [5000 * x + 500 for x in std.range(4, 10)],
    eval_steps: [55500],
    eval_section: "val",
}
