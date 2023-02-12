
# load LLM
# load Text dataset
# de-unicode the dataset
# split into sentences/tts-style chunks
# normalize numbers(? does G2P do this for us?)
# remove samples with URLs
# apply G2P
# apply DeepMoji # note model has only seen max 140 chars per sample during training
# get sequence ids
# get word ids (value)
# get phoneme ids (value)
# get character ids (value)
# get word ids (position)
# get phoneme ids (position)
# get character ids (position)
# apply masking
# ensure no word ids exist in both input and target at the same time
