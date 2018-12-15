#-*-coding:utf-8-*-
import numpy as np
import logging
import sys
import os
import h5py
import argparse
from model_factory import ModelFactory
from data_process import Vocab, DataGenerator, ModelParam

np.random.seed(1337)

log = logging.getLogger("output")
logging.basicConfig(level=logging.INFO,
				format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
				datefmt='%a, %d %b %Y %H:%M:%S',
				filename='log.txt',
				filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [line:%(lineno)d] %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

parser = argparse.ArgumentParser(description='Dynamic-Clip Attention')
parser.add_argument('-t','--task',type=str,default="wikiqa",help='task: trecqa or wikiqa')
parser.add_argument('-m',"--model",type=str,default="listwise",help='model: listwise, k_max or k_threhold')
parser.add_argument('-e_d',"--embeb_dim",type=int,default=100,help='dimension of the hidden layer')
parser.add_argument('-d',"--hidden_dim",type=int,default=300,help='dimension of the hidden layer')
parser.add_argument('-e',"--epoch",type=int,default=10,help='max epoch')
parser.add_argument('-l',"--lr",type=float,default=0.001,help='learning rate')
parser.add_argument('-k_q',"--k_value_ques",type=float,default=10,help='k value in k_max or k_threshold for question attention')
parser.add_argument('-k_a',"--k_value_ans",type=float,default=10,help='k value in k_max or k_threshold for answer attention')
parser.add_argument('-b',"--batch_size",type=int,default=3,help='batch_size')
parser.add_argument('-p',"--pre_train",type=int,default=5,help='pre_train')

global_mark = "wikiqa" + "_" +"listwise" 

def result_log(str):
	ff = open("./result_note.txt",'a')
	ff.write(str+"\n")
	ff.close()


def specific_load_epoch(model,epoch,prefix):  #load model from model folder
	assert os.path.exists('model/'+prefix+'_weights_epoch_%s.h5' %
						  epoch), 'Weights at epoch %s not found' % epoch
	filename = 'model/'+prefix+'_weights_epoch_%s.h5' % epoch
	h5_file = h5py.File(filename,'r')
	weight = []
	for i in range(len(h5_file.keys())):
		weight.append(h5_file['weight'+str(i)][:])
	model.set_weights(weight)


def specific_save_epoch(model,epoch,prefix,global_mark_copy):  #save
	if not os.path.exists('model/'):
		os.makedirs('model/')
	filename = 'model/'+prefix+'_weights_epoch_%s.h5' % (str(epoch)+"_"+global_mark_copy)
	h5_file = h5py.File(filename,'w')
	weight = model.get_weights()
	for i in range(len(weight)):
		h5_file.create_dataset('weight'+str(i),data=weight[i])
	h5_file.close()

def prog_bar(so_far, total, n_bars=20):
	n_complete = int(so_far * n_bars / total)
	if n_complete >= n_bars - 1:
		sys.stderr.write('\r[' + '=' * n_bars + ']')
	else:
		s = '\r[' + '=' * (n_complete - 1) + '>' + '.' * \
			(n_bars - n_complete) + ']'
		sys.stderr.write(s)

def task_data_ready(task,model_param):  #get all pre processed data
	if task == "wikiqa":
		vocab_all = Vocab("./data/wikiqa/vocab_wiki.txt", max_size=80000)
		data_generator = DataGenerator(vocab_all, model_param,"./data/wikiqa/wiki_answer_train.pkl") 
		embedding_file = "./data/wikiqa/wikiqa_glovec.txt"
		dev_data = data_generator.EvaluateGenerate("./data/wikiqa/wiki_dev.pkl")
		test_data = data_generator.EvaluateGenerate("./data/wikiqa/wiki_test.pkl")	
	elif task == "trecqa":
		vocab_all = Vocab("./data/trecqa/vocab_trec.txt", max_size=80000)
		data_generator = DataGenerator(vocab_all, model_param,"./data/trecqa/trec_answer_train.pkl")
		embedding_file = "./data/trecqa/trecqa_glovec.txt"
		dev_data = data_generator.EvaluateGenerate("./data/trecqa/trec_dev.pkl")
		test_data = data_generator.EvaluateGenerate("./data/trecqa/trec_test.pkl")
	elif task == "insqa":
		vocab_all = Vocab("./data/insqa/vocab_insqa.txt", max_size=80000)
		data_generator = DataGenerator(vocab_all, model_param, "./data/insqa/insqa_answer_train.pkl")
		embedding_file = "./data/insqa/insqa_glovec.txt"
		dev_data = data_generator.EvaluateGenerate("./data/insqa/insqa_dev.pkl")
		test_data = data_generator.EvaluateGenerate("./data/insqa/insqa_test.pkl")
	elif task == "selqa":
		vocab_all = Vocab("./data/selqa/vocab_selqa.txt", max_size=80000)
		data_generator = DataGenerator(vocab_all, model_param, "./data/selqa/selqa_answer_train.pkl")
		embedding_file = "./data/selqa/selqa_glovec.txt"
		dev_data = data_generator.EvaluateGenerate("./data/selqa/selqa_dev.pkl")
		test_data = data_generator.EvaluateGenerate("./data/selqa/selqa_test.pkl")
	return vocab_all,data_generator,embedding_file,dev_data,test_data



def main(args):
	global_mark = args.task + "_" + args.model
	print str(args.pre_train)+" model"
	if args.task == "wikiqa":  #ModelParam is just a tuple format, random_size means the neg_answ pool size
		model_param = ModelParam(hidden_dim=args.hidden_dim, enc_timesteps=12, dec_timesteps=50, batch_size=args.batch_size, random_size=15, lr=args.lr, k_value_ques=args.k_value_ques,k_value_ans=args.k_value_ans)
	elif args.task == "trecqa":
		model_param = ModelParam(hidden_dim=args.hidden_dim, enc_timesteps=12, dec_timesteps=50, batch_size=args.batch_size, random_size=15, lr=args.lr, k_value_ques=args.k_value_ques,k_value_ans=args.k_value_ans)
	elif args.task == "insqa":
		model_param = ModelParam(hidden_dim=args.hidden_dim, enc_timesteps=12, dec_timesteps=50, batch_size=args.batch_size, random_size=50, lr=args.lr, k_value_ques=args.k_value_ques,k_value_ans=args.k_value_ans)
	elif args.task == "selqa":
		model_param = ModelParam(hidden_dim=args.hidden_dim, enc_timesteps=12, dec_timesteps=50, batch_size=args.batch_size, random_size=15, lr=args.lr, k_value_ques=args.k_value_ques,k_value_ans=args.k_value_ans)

	logging.info(model_param.__str__())
	vocab_all,data_generator,embedding_file,dev_data,test_data = task_data_ready(args.task,model_param)
	
	train_model, predict_model = ModelFactory.get_model(model_param, embedding_file, vocab_all.NumIds(),model_type=args.model) 

	#after prepared model format
	def data_evaluate(epoch,small_evaluate_data,flag):
		c_1_j = 0		
		c_2_j = 0
		Top1 = 0.0
		no_answer = 0.0
		for i, d in enumerate(small_evaluate_data.values()):
			prog_bar(i, len(small_evaluate_data))
			question = d["question"]
			answers = d["answer"]
			question_len = d["ques_len"]
			ans_len = d["ans_len"]
			sims = predict_model.predict([question,answers,question_len,ans_len],batch_size=len(question))
			sims = sims[:,0]
			rank_index = np.argsort(sims).tolist()[::-1]
			score = 0.0
			count = 0.0
			if np.sum(d["label"])>0:
				for i in range(1,len(sims)+1):
					if d["label"][rank_index[i-1]] == 1:
						count += 1
						score += count / i
				for i in range(1,len(sims)+1):
					if d["label"][rank_index[i-1]] == 1:
						c_2_j += 1/float(i)
						break
				if d["label"][rank_index[0]] == 1:
					Top1 += 1
				c_1_j += score / count
			else:
				no_answer += 1
				
		Top1 = Top1 / float(len(small_evaluate_data)-no_answer)
		MAP = c_1_j / float(len(small_evaluate_data)-no_answer)
		MRR = c_2_j / float(len(small_evaluate_data)-no_answer)
		print("no_answer:",no_answer)
		logging.info(global_mark + " evaluate on "+ flag +" data at epoch "+str(epoch)+' MAP: %f' % MAP)
		logging.info(" evaluate on "+ flag + " data at epoch "+str(epoch)+' MRR: %f' % MRR)
		logging.info(" evaluate on " + flag + " data at epoch " + str(epoch) + ' Top@1: %f' % Top1)
		result_log(global_mark + " evaluate on " + flag + " data at epoch " + str(epoch) + ' Top@1: %f' % Top1)
		result_log(global_mark+" evaluate on "+ flag + " data at epoch "+str(epoch)+' MAP: %f' % MAP)
		result_log(global_mark+" evaluate on "+ flag + " data at epoch "+str(epoch)+' MRR: %f' % MRR)
		return Top1,MAP,MRR

	if args.model != "listwise":
		reload_epoch = args.pre_train
		assert os.path.exists('model/train_weights_epoch_%s.h5' % (str(reload_epoch)+"_"+args.task+"_listwise")), "please pre-train listwise approach"
		specific_load_epoch(train_model,str(reload_epoch)+"_"+args.task+"_listwise","train")	
	best_epoch = 0
	best_map = 0
	score_list = []
	for i in range(1,args.epoch+1):
		if args.task == "wikiqa":
			train_filename = "./data/wikiqa/wiki_train.pkl"		
			questions, answers, label, question_len ,answer_len = data_generator.wikiQaGenerate(train_filename,"basic")
		elif args.task == "trecqa":
			train_filename = "./data/trecqa/trec_train.pkl"		
			questions, answers, label, question_len ,answer_len = data_generator.trecQaGenerate(train_filename,"basic")
		elif args.task == "insqa":
			train_filename = "./data/insqa/insqa_train.pkl"
			questions, answers, label, question_len, answer_len = data_generator.trecQaGenerate(train_filename, "basic")
		elif args.task == "selqa":
			train_filename = "./data/selqa/selqa_train.pkl"
			questions, answers, label, question_len, answer_len = data_generator.trecQaGenerate(train_filename, "basic")

		logging.info('Fitting epoch %d' % i)

		train_model.fit([questions, answers,question_len,answer_len], label,nb_epoch=1, batch_size=model_param.batch_size, validation_split=0, verbose=1,shuffle=True)

		####evaluate 
		dev_top1, dev_map,dev_mrr = data_evaluate(i,dev_data,"dev")
		test_top1, test_map,test_mrr = data_evaluate(i,test_data,"test")
		if dev_map > best_map:
			best_map = dev_map
			best_epoch = i
			score_list = [dev_map,dev_mrr,test_map,test_mrr,dev_top1,test_top1]
	#######best result
	logging.info("best model at epoch "+str(best_epoch))
	logging.info("the dev score of best model: MAP_"+str(score_list[0])+" MRR_"+str(score_list[1])+"Top1_"+str(score_list[4]))
	logging.info("the test score of best model: MAP_"+str(score_list[2])+" MRR_"+str(score_list[3])+"Top1_"+str(score_list[5]))

if __name__ == "__main__":
	args = parser.parse_args()
	main(args)
