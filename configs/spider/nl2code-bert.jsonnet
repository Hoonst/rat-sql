local _base = import 'nl2code-base.libsonnet';
local _output_from = true;
local _fs = 2;

function(args) _base(output_from=_output_from, data_path=args.data_path) + {
    local data_path = args.data_path,
    
    local lr_s = '%0.1e' % args.lr,
    local plm_lr_s = '%0.1e' % args.plm_lr,
    local end_lr_s = if args.end_lr == 0 then '0e0' else '%0.1e' % args.end_lr,

    # local base_plm_enc_size = if args.plm_version == "bert-large-uncased-whole-word-masking" or args.plm_version == "google/electra-large-discriminator" then 1024 else 768,
    local base_plm_enc_size = 1024,
    local enc_size =  base_plm_enc_size,
    local loss_s = args.loss,
    local qv_link = args.qv_link,
    local dist_relation = args.dist_relation,
    local orthog = args.use_orthogonal,
    local orth_init = args.use_orth_init,
    local bi_way = args.bi_way,
    local bi_match = args.bi_match,
    local dp_link = args.dp_link,
    local plm = if args.plm_version == "bert-large-uncased-whole-word-masking" then 'bert' else if args.plm_version =="google/electra-large-discriminator" then "electra" else if args.plm_version =="microsoft/deberta-v3-large" then "debertav3" else if args.plm_version=="microsoft/deberta-large" then "debertav1",
    local att_seq = args.att_seq,
    local layers = args.num_layers,
    
    model_name: 'bs=%(bs)d,qv_link=%(qv_link)s,dist=%(dist_relation)s,orthog=%(orthog)s,orth_init=%(orth_init)s,bi_way=%(bi_way)s,bi_match=%(bi_match)s,dp_link=%(dp_link)s,plm=%(plm)s,att_seq=%(att_seq)s,layers=%(layers)s' % (args + {
        qv_link: qv_link,
        dist_relation: dist_relation,
        orthog: orthog,
        orth_init: orth_init,
        bi_way: bi_way,
        bi_match: bi_match,
        dp_link: dp_link,
        plm: plm,
        att_seq: att_seq,
        layers: layers,
    }),
    
    model+: {
        encoder+: {
            name: 'spider-bert',
            batch_encs_update:: false,
            question_encoder:: null,
            column_encoder:: null,
            table_encoder:: null,
            dropout:: null,
            update_config+:  {
                name: 'relational_transformer',
                num_layers: args.num_layers,
                num_heads: 8,
                sc_link: args.sc_link,
                cv_link: args.cv_link,
                qv_link: args.qv_link,
                dp_link: args.dp_link,
                att_seq: args.att_seq,
                dist_relation: args.dist_relation,
                orth_init: args.use_orth_init,
                bi_match: args.bi_match,
                bi_way: args.bi_way,
            },
            enc_qv_link: args.qv_link,
            use_orthogonal: args.use_orthogonal,
            summarize_header: args.summarize_header,
            use_column_type: args.use_column_type,
            plm_version: args.plm_version,
            plm_token_type: args.plm_token_type,
            qv_link: args.qv_link,
            dp_link: args.dp_link,
            top_k_learnable:: null,
            word_emb_size:: null,
        },
        encoder_preproc+: {
            word_emb:: null,
            min_freq:: null,
            max_count:: null,
            db_path: data_path + "database",
            compute_sc_link: args.sc_link,
            compute_cv_link: args.cv_link,
            compute_dp_link: args.dp_link,
            fix_issue_16_primary_keys: true,
            plm_version: args.plm_version,
            count_tokens_in_word_emb_for_vocab:: null,
            save_path: data_path + 'nl2code,output_from=%s,fs=%d,emb=bert,cvlink,dp_link=%s' % [_output_from, _fs, args.dp_link],
        },
        decoder_preproc+: {
            grammar+: {
                end_with_from: args.end_with_from,
                clause_order: args.clause_order,
                infer_from_conditions: true,
                factorize_sketch: _fs,
            },
            save_path: data_path + 'nl2code,output_from=%s,fs=%d,emb=bert,cvlink,dp_link=%s' % [_output_from, _fs, args.dp_link],

            compute_sc_link:: null,
            compute_cv_link:: null,
            compute_dp_link:: null,
            db_path:: null,
            fix_issue_16_primary_keys:: null,
            plm_version:: null,
        },

        decoder+: {
            name: 'NL2Code',
            dropout: 0.20687225956012834,
            desc_attn: 'mha',
            enc_recurrent_size: enc_size,
            recurrent_size : args.decoder_hidden_size,
            loss_type: args.loss,
            use_align_mat: args.use_align_mat,
            use_align_loss: args.use_align_loss,
        }
    },

    train+: {
        batch_size: args.bs,
        num_batch_accumulated: args.num_batch_accumulated,
        clip_grad: 1,

        model_seed: args.att,
        data_seed:  args.att,
        init_seed:  args.att,
        
        max_steps: args.max_steps,
    },

    optimizer: {
        name: 'bertAdamw',
        lr: 0.0,
        bert_lr: 0.0,
    },

    lr_scheduler+: {
        name: 'bert_warmup_polynomial_group',
        start_lrs: [args.lr, args.plm_lr],
        end_lr: args.end_lr,
        num_warmup_steps: $.train.max_steps / 8,
    },

    log: {
        reopen_to_flush: true,
    }
}