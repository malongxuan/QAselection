#-*-coding:utf-8 -*-
# from __future__ import print_function
import sys
import numpy as np
import random
from collections import namedtuple #namedtuple(typename, field_names, verbose=False, rename=False) 
								   #Returns a new subclass of tuple with named fields.
import pickle
random.seed(1337)

ModelParam = namedtuple("ModelParam","hidden_dim,enc_timesteps,dec_timesteps,batch_size,random_size,k_value_ques,k_value_ans,lr")

UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
class Vocab(object):  #has _word_to_id  _id_to_word  _count(how many ID)
	def __init__(self, vocab_file, max_size):
		self._word_to_id = {}
		self._id_to_word = {}
		self._count = 0
		before_list = [PAD_TOKEN]
		for word in before_list: #connect word with id(_count),  _word_to_id[PAD_TOKEN] = 0,  _id_to_word[0]=PAD_TOKEN
				self.CreateWord(word)
		with open(vocab_file, 'r') as vocab_f:
			for line in vocab_f:
				pieces = line.split()
				if len(pieces) != 2:  #str(index)+" "+word+"\n"
					sys.stderr.write('Bad line: %s\n' % line)
					continue
				if pieces[1] in self._word_to_id:  #because vocab is dict, should not have duplicate
					raise ValueError('Duplicated word: %s.' % pieces[1])
				self._word_to_id[pieces[1]] = self._count  #connect all words in vocab with id(_count)
				self._id_to_word[self._count] = pieces[1]
				self._count += 1
				if self._count > max_size-1:  #only use limit words
					sys.stderr.write('Too many words: >%d.' % max_size)
					break
	def WordToId(self, word):
		if word not in self._word_to_id:
			return self._word_to_id[UNKNOWN_TOKEN]
		return self._word_to_id[word]

	def IdToWord(self, word_id):
		if word_id not in self._id_to_word:
			raise ValueError('id not found in vocab: %d.' % word_id)
		return self._id_to_word[word_id]

	def NumIds(self):
		return self._count

	def CreateWord(self,word):
		if word not in self._word_to_id: #connect word with id(_count), _word_to_id[PAD_TOKEN] = 0, _id_to_word[0]=PAD_TOKEN
			self._word_to_id[word] = self._count
			self._id_to_word[self._count] = word
			self._count += 1
	def Revert(self,indices): #get word of sentence by id, return word or 'X'
		vocab = self._id_to_word
		return [vocab.get(i, 'X') for i in indices] #for D={}, D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.
	def Encode(self,indices):  #get id of sentence by word, return id or 'nonum'
		vocab = self._word_to_id
		return [vocab.get(i, 'nonum') for i in indices]##D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.


class DataGenerator(object):
	"""Dataset class
		vocab_all = Vocab("./data/wikiqa/vocab_wiki.txt", max_size=80000)
		data_generator = DataGenerator(vocab_all, model_param, "./data/wikiqa/wiki_answer_train.pkl") 
		ModelParam(hidden_dim=args.hidden_dim, enc_timesteps=25, dec_timesteps=90, batch_size=args.batch_size, 
					random_size=15, lr=args.lr, k_value_ques=args.k_value_ques,k_value_ans=args.k_value_ans)
	"""
	def __init__(self,vocab,model_param,answer_file = ""):
		self.vocab = vocab  #str(index)+" "+word+"\n"
		self.param = model_param
		self.batch_size = self.param.batch_size
		self.corpus_amount = 0
		if answer_file != "":
			self.answers = pickle.load(open(answer_file,'rb'))
		#for insqa:print(len(answers))  # 24981
		#print(type(answers))  # <class 'dict'>

	def padq(self, data):
		return self.pad(data, self.param.enc_timesteps)

	def pada(self, data):
		return self.pad(data, self.param.dec_timesteps)

	def pad(self, data, len=None):
		from keras.preprocessing.sequence import pad_sequences
		return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

	def wikiQaGenerate(self,filename,flag="basic"):
	#train_filename = "./data/wikiqa/wiki_train.pkl"
	#questions, answers, label, question_len ,answer_len = data_generator.wikiQaGenerate(train_filename,"basic")
		data = pickle.load(open(filename,'r')) #list 20360
		question_dic = {}
		question = list()
		answer = list()
		label = list()
		question_len = list()
		answer_len = list()
		answer_size = list()
		for item in data:	#item is ([1, 2, 3, 4, 5, 6], [7, 8, 9, 3, 10, 11, 12, 13, 3, 14], 0)
							#next item is ([1, 2, 3, 4, 5, 6], [15, 16, 17, 18, 19, 20, 21, 22], 0)
							#str(item[0]) is "[1, 2, 3, 4, 5, 6]",item[0] is list
			question_dic.setdefault(str(item[0]),{})  #question_dic[str]:{}
			question_dic[str(item[0])].setdefault("question",[])  ##question_dic[str]:{"question":[]}
			question_dic[str(item[0])].setdefault("answer",[])    ##question_dic[str]:{"question":[],"answer":[]}
			question_dic[str(item[0])].setdefault("label",[])     ##question_dic[str]:{"question":[],"answer":[],"label":[]}
			question_dic[str(item[0])]["question"].append(item[0])##question_dic[str]:{"question":[num num num ....]}
			question_dic[str(item[0])]["answer"].append(item[1])
			question_dic[str(item[0])]["label"].append(item[2])	#question_dic[str]:{"question":[[num num....],[num num....],...],
																#					"answer":[[num1 num2...],[num3 num4...],...],
																#					"label":[0,1,...]}
		delCount = 0
		for key in question_dic.keys():
			question_dic[key]["question"] = question_dic[key]["question"]	#?????
			question_dic[key]["answer"] = question_dic[key]["answer"]		#?????
			if sum(question_dic[key]["label"]) == 0:
				delCount += 1
				del(question_dic[key])   #remove no groundtrue
		for item in question_dic.values():
			good_answer = [item["answer"][i] for i in range(len(item["question"])) if item["label"][i] == 1] 
			good_length = len(good_answer)  #how many pos
			bad_answer = [item["answer"][i] for i in range(len(item["question"])) if item["label"][i] == 0] 
			trash_sample = self.param.random_size
			if len(item["answer"]) >= self.param.random_size:
				good_answer.extend(random.sample(bad_answer,self.param.random_size - good_length))
				temp_answer = good_answer  #get random_size pool
				temp_label = [1 / float(sum(item["label"])) for i in range(good_length)]   #for Loss function??????
				temp_label.extend([0.0 for i in range(self.param.random_size-good_length)])
			else:
				temp_answer = item["answer"]  #get all_answer pool
				temp_answer.extend(random.sample(self.answers.values(), self.param.random_size-len(item["question"])))
											#get random_size pool using wiki_answer_train.pkl
				temp_label = [ll / float(sum(item["label"])) for ll in item["label"]]   ##??????????????
				temp_label.extend([0.0 for i in range(self.param.random_size-len(item["question"]))])
				trash_sample = len(item["question"])   #trash_sample set to num of ("QA pair")
			label.append(temp_label)
			answer.append(self.pada(temp_answer))
			length = [1 for i in range(len(item["question"][0]))]

			ans_length = [[1 for i in range(len(single_ans))] for single_ans in temp_answer]
			answer_len.append(self.pada(ans_length))   #??????? pada and padq input is data ?????
			question_len += [self.padq([length])[0]]
			question += [self.padq([item["question"][0]])[0]]
			answer_size += [[1 for i in range(self.param.random_size) if i < trash_sample] + [0 for i in range(self.param.random_size-trash_sample)] ]

		question = np.array(question)
		answer = np.array(answer)
		label = np.array(label)
		#print("len(question_len):",len(question_len))		#('len(question_len):', 873)
		#print("shape of question_len[0]",question_len[0].shape)#('shape of question_len[0]', (25,))
		#print(question_len[0])	#[1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

		question_len = np.array(question_len)
		answer_len = np.array(answer_len)
		answer_size = np.array(answer_size)

		print question.shape #(873, 25)
		print answer.shape  #(873, 15, 90)
		print label.shape   #(873, 15)

		if flag == "size":
			return question,answer,label,question_len,answer_len,answer_size
		return question,answer,label,question_len,answer_len

	def trecQaGenerate(self,filename,flag="basic"):
		data = pickle.load(open(filename,'r'))
		question_dic = {}
		question = list()
		answer = list()
		label = list()
		question_len = list()
		answer_len = list()
		answer_size = list()
		for item in data:
			question_dic.setdefault(str(item[0]),{})
			question_dic[str(item[0])].setdefault("question",[])
			question_dic[str(item[0])].setdefault("answer",[])
			question_dic[str(item[0])].setdefault("label",[])
			question_dic[str(item[0])]["question"].append(item[0])
			question_dic[str(item[0])]["answer"].append(item[1])
			question_dic[str(item[0])]["label"].append(item[2])
		delCount = 0
		for key in question_dic.keys():
			question_dic[key]["question"] = question_dic[key]["question"]
			question_dic[key]["answer"] = question_dic[key]["answer"]
			if sum(question_dic[key]["label"]) == 0:
				delCount += 1
				del(question_dic[key])
		for item in question_dic.values():
			good_answer = [item["answer"][i] for i in range(len(item["question"])) if item["label"][i] == 1] 
			good_length = len(good_answer)
			if good_length >= self.param.random_size/2:
				good_answer = random.sample(good_answer,self.param.random_size/2)
				good_length = len(good_answer)
			bad_answer = [item["answer"][i] for i in range(len(item["question"])) if item["label"][i] == 0] 
			trash_sample = self.param.random_size
			if len(bad_answer) >= self.param.random_size - good_length:
				good_answer.extend(random.sample(bad_answer,self.param.random_size - good_length))
				temp_answer = good_answer
				temp_label = [1 / float(good_length) for i in range(good_length)]
				temp_label.extend([0.0 for i in range(self.param.random_size-good_length)])
			else:
				temp_answer = good_answer + bad_answer
				trash_sample = len(temp_answer)
				temp_answer.extend(random.sample(self.answers.values(), self.param.random_size-len(temp_answer)))
				temp_label = [1 / float(len(good_answer)) for i in range(len(good_answer))]
				temp_label.extend([0.0 for i in range(self.param.random_size-len(good_answer))])
			
			label.append(temp_label)
			answer.append(self.pada(temp_answer))
			length = [1 for i in range(len(item["question"][0]))]

			ans_length = [[1 for i in range(len(single_ans))] for single_ans in temp_answer]
			answer_len.append(self.pada(ans_length))
			question_len += [self.padq([length])[0]]
			question += [self.padq([item["question"][0]])[0]]
			answer_size += [[1 for i in range(self.param.random_size) if i < trash_sample] + [0 for i in range(self.param.random_size-trash_sample)] ]

		question = np.array(question)
		answer = np.array(answer)
		label = np.array(label)
		question_len = np.array(question_len)
		answer_len = np.array(answer_len)
		answer_size = np.array(answer_size)
		print question.shape
		print answer.shape
		print label.shape
		if flag == "size":
			return question,answer,label,question_len,answer_len,answer_size
		return question,answer,label,question_len,answer_len

	def EvaluateGenerate(self,filename):
		data = pickle.load(open(filename,'r'))
		question_dic = {}
		for item in data:
			question_dic.setdefault(str(item[0]),{})
			question_dic[str(item[0])].setdefault("question",[])
			question_dic[str(item[0])].setdefault("answer",[])
			question_dic[str(item[0])].setdefault("label",[])
			question_dic[str(item[0])]["question"].append(item[0])
			question_dic[str(item[0])]["answer"].append(item[1])
			question_dic[str(item[0])]["label"].append(item[2])
		delCount = 0
		for key in question_dic.keys():
			question_dic[key]["question"] = self.padq(question_dic[key]["question"])
			question_dic[key]["answer"] = self.pada(question_dic[key]["answer"])
			question_dic[key]["ques_len"] = self.padq([[1 for i in range(len(single_que))] for single_que in question_dic[key]["question"] ])
			question_dic[key]["ans_len"] = self.pada([[1 for i in range(len(single_ans))] for single_ans in question_dic[key]["answer"] ])

			if sum(question_dic[key]["label"]) == 0:
				delCount += 1
				del(question_dic[key])
		print delCount            #170   390   raw
		print len(question_dic)   #126   243   reduced
		return question_dic
	"""
		pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)
		Pads each sequence to the same length (length of the longest sequence).
		
		If maxlen is provided, any sequence longer
		than maxlen is truncated to maxlen.
		Truncation happens off either the beginning (default) or
		the end of the sequence.
		
		Supports post-padding and pre-padding (default).
		
		Arguments
			sequences: list of lists where each element is a sequence
			maxlen: int, maximum length
			dtype: type to cast the resulting sequence.
			padding: 'pre' or 'post', pad either before or after each sequence.
			truncating: 'pre' or 'post', remove values from sequences larger than
				maxlen either in the beginning or in the end of the sequence
			value: float, value to pad the sequences to the desired value.
		
		Returns
			x: numpy array with dimensions (number_of_sequences, maxlen)
		
		Raises
			ValueError: in case of invalid values for `truncating` or `padding`,
				or in case of invalid shape for a `sequences` entry.
	"""