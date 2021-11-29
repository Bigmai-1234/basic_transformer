from transformers import BertTokenizer
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained('../bert_model/chinese_L-12_H-768_A-12', do_lower_case=True)
max_length_test = 30
test_sentence = '这是一个例子，我不知道为什么要用这个例子。'

# add special tokens
test_sentence_with_special_tokens = '[CLS]' + test_sentence + '[SEP]'
tokenized = tokenizer.tokenize(test_sentence_with_special_tokens)
print('tokenized', tokenized)
# convert tokens to ids in WordPiece
input_ids = tokenizer.convert_tokens_to_ids(tokenized)
# precalculation of pad length, so that we can reuse it later on
padding_length = max_length_test - len(input_ids)
# input_ids = tf.keras.preprocessing.sequence.pad_sequences([input_ids], max_length_test)
# attention should focus just on sequence with non padded tokens
attention_mask = [1] * len(input_ids)
# # do not focus attention on padded tokens
attention_mask = attention_mask + ([0] * padding_length)
# map tokens to WordPiece dictionary and add pad token for those text shorter than our max length
input_ids = input_ids + ([0] * padding_length)
# attention_mask = tf.keras.preprocessing.sequence.pad_sequences([attention_mask], max_length_test)
# token types, needed for example for question answering, for our purpose we will just set 0 as we have just one sequence
token_type_ids = [0] * max_length_test

bert_input = {
    "token_ids": input_ids,
    "token_type_ids": token_type_ids,
    "attention_mask": attention_mask
}
print(bert_input)

bert_input = tokenizer.encode_plus(
    test_sentence,
    add_special_tokens=True,  # add [CLS], [SEP]
    max_length=max_length_test,  # max length of the text that can go to BERT
    pad_to_max_length=True,  # add [PAD] tokens
    return_attention_mask=True,  # add attention mask to not focus on pad tokens
)

print('encoded', bert_input)
