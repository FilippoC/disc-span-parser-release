cmd ={
	#"--word-dropout": "",
	"--min-word-freq": "2",
	"--word-embs-dim": "300",
	"--char-embs": "",
	"--char-embs-dim": "64",
	"--char-lstm-dim": "100",
        "--train": "/mnt/beegfs/home/corro/repos/disc_span_parser/data/negra/train.export",
        "--dev": "/mnt/beegfs/home/corro/repos/disc_span_parser/data/negra/dev.export",
        #"--test": "/mnt/beegfs/home/corro/repos/disc_span_parser/data/negra/test.export",
        "--model": "model",
        "--device": "cuda:0",
        "--epochs": "200",
        "--optim": "adam",
        "--optim-adam-b1": "0.9",
        "--optim-adam-b2": "0.9",
        "--optim-adam-eps": "1e-12",
        "--char-lstm-boundaries": "",
	"--dropout-char-lstm-input": "0.0",
	"--label-proj-dim": "100",
	"--span-proj-dim": "500",
	"--batch": "5000",
	"--batch-clusters": "32",
	"--mean-loss": "",
}

cmd_options = [
[
	("lstm400", {"--lstm-dim": "400"}),
	("lstm600", {"--lstm-dim": "600"}),
	("lstm800", {"--lstm-dim": "800"}),
],
[
	("stacks1_layers3", {
		"--lstm-layers": "3",
		"--lstm-stacks": "1"
	}),
	("stacks2_layers1", {
		"--lstm-layers": "1",
		"--lstm-stacks": "2"
	}),
	("stacks2_layers2", {
		"--lstm-layers": "2",
		"--lstm-stacks": "2"
	}),
],
[
	("dropout033", {
		"--dropout-features": "0.33",
		"--lstm-dropout": "0.33",
		"--mlp-dropout": "0.33",
	}),
	("dropout05", {
		"--dropout-features": "0.33",
		"--lstm-dropout": "0.5",
		"--mlp-dropout": "0.5",
	}),
],
[
	("tagger", {"--tagger": ""}),
	("notagger", {})
],
[
	("char_lim20", {"--max-word-len": "20"}),
	("no_char_lim", {})
],
[
	("lr_exp", {
        	"--optim-lr": "2e-3",
		"--optim-lr-scheduler": "exponential",
		"--optim-lr-scheduler-step": "5000",
		"--optim-lr-scheduler-decay": "0.75",
	}),
],
[
	("rand_embs", {
		"--word-embs": "",
	}),
	#("pretrained_embs", {
	#	"--pretrained-word-embs": "",
	#	"--pretrained-word-embs-finetune": "",
	#	"--pretrained-word-embs-path": "/mnt/beegfs/home/corro/interpretable-nn/embeddings/glove_german.txt",
	#	"--pretrained-word-embs-dim": "300"
	#}),
],
]

