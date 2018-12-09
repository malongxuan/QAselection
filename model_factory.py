#-*-coding:utf-8-*-
import theano 
import theano.tensor as T 
from keras.layers import LSTM, Dense, Activation, Dropout, Input, merge, RepeatVector, Merge, Lambda ,Flatten, BatchNormalization,Permute
from keras.layers.convolutional import Convolution1D
from keras.models import Model
from keras.optimizers import RMSprop,Adamax,SGD,Adam
from keras.layers import Embedding
from keras import backend as K
from keras.layers.wrappers import TimeDistributed
import numpy as np
np.random.seed(1337)

class ModelFactory(object):
	@staticmethod
	def get_listwise_model(model_param, embedding_file, vocab_size):
		def get_core_model(model_param, embedding_file, vocab_size):
			class _Attention(object):#co
				def __init__(self, ques_length, answer_length, nr_hidden):
					self.ques_length = ques_length
					self.answer_length = answer_length
				def __call__(self, sent1, sent2, reverse = False):
					def _outer(AB):
						att_ji = K.batch_dot(AB[1], K.permute_dimensions(AB[0], (0, 2, 1)))
						return K.permute_dimensions(att_ji,(0, 2, 1))
					if reverse:
						return merge(
							[sent2, sent1],
							mode=_outer,
							output_shape=(self.answer_length, self.ques_length))
					else:
						return merge(
							[sent1, sent2],
							mode=_outer,
							output_shape=(self.ques_length, self.answer_length))
			class _SoftAlignment_mask(object):#for first co and intra, output len*hidden
				def __init__(self, nr_hidden):
					self.nr_hidden = nr_hidden
				def __call__(self, sentence, attention, ques_len, max_length):
					def _normalize_attention(attmat):
						att = attmat[0]
						mat = attmat[1]
						ques_len = attmat[2]
						att = K.permute_dimensions(att,(0, 2, 1))
						# 3d softmax
						e = K.exp(att - K.max(att, axis=-1, keepdims=True))
						g = e * ques_len
						
						if max_length>40:
							bound = -10
						elif max_length<35:
							bound = -25
						else:
							bound = -10
						k_max_e = K.T.set_subtensor(g[K.T.arange(g.shape[0]).dimshuffle(0,'x','x'),
													K.T.arange(g.shape[1]).dimshuffle('x',0,'x'), 
													K.T.argsort(g)[:,:,:bound]],
													0.0)

						s = K.sum(k_max_e,axis=-1,keepdims=True)
						sm_att = k_max_e / s
						return K.batch_dot(sm_att, mat)	
					return merge([attention, sentence, ques_len], mode=_normalize_attention,
								  output_shape=(max_length, self.nr_hidden)) 

			class _SoftAlignment(object):#for second co, output len*hidden
				def __init__(self, nr_hidden):
					self.nr_hidden = nr_hidden
				def __call__(self, sentence, attention, w_att, max_length):
					def _normalize_attention(attmat):
						att = attmat[0]
						mat = attmat[1]
						w_att = attmat[2]
						#att = att-w_att
						att = K.permute_dimensions(att,(0, 2, 1))
						w_att = K.permute_dimensions(w_att,(0, 2, 1))
						# 3d softmax
						g = K.exp(att - K.max(att, axis=-1, keepdims=True))
						w_g = K.exp(w_att - K.max(w_att, axis=-1, keepdims=True))
						s = K.sum(g,axis=-1,keepdims=True)
						w_s = K.sum(w_g,axis=-1,keepdims=True)
						sm_att = g / s
						w_sm_att = w_g / w_s
						k_threshold_e = T.switch(K.less(sm_att,w_sm_att), 0.0, sm_att)
						new_s = T.clip(T.sum(k_threshold_e,axis=-1,keepdims=True),0.00001,1024)
						new_sm_att = k_threshold_e / new_s
						return K.batch_dot(new_sm_att, mat)
					return merge([attention, sentence, w_att], mode=_normalize_attention,
								  output_shape=(max_length, self.nr_hidden)) 
			class _max_soft(object):#for co and intra, output 1*hidden
				def __init__(self, nr_hidden):
					self.nr_hidden = nr_hidden
				def __call__(self, sentence, attention, input_length, output_length=1):
					def _normalize_attention(attmat):
						att = attmat[0]
						mat = attmat[1]
						e = K.max(att, axis=-1, keepdims=True)
						# 3d softmax
						e = K.permute_dimensions(e,(0, 2, 1))
						g = K.exp(e - K.max(e, axis=-1, keepdims=True))
						
						if input_length>40:
							bound = -25
						elif input_length<35:
							bound = -10
						else:
							bound = -10
						k_max_e = K.T.set_subtensor(g[K.T.arange(g.shape[0]).dimshuffle(0,'x','x'),
													K.T.arange(g.shape[1]).dimshuffle('x',0,'x'), 
													K.T.argsort(g)[:,:,:bound]],
													0.0)
						
						s = K.sum(k_max_e,axis=-1,keepdims=True)
						sm_att = k_max_e / s
						return K.batch_dot(sm_att, mat)	
					return merge([attention, sentence], mode=_normalize_attention,
								  output_shape=(output_length, self.nr_hidden)) 
			class _mean_soft(object):#for co and intra, output 1*hidden
				def __init__(self, nr_hidden):
					self.nr_hidden = nr_hidden
				def __call__(self, sentence, attention, input_length, output_length=1):
					def _normalize_attention(attmat):
						att = attmat[0]
						mat = attmat[1]
						e = K.mean(att, axis=-1, keepdims=True)
						# 3d softmax
						e = K.permute_dimensions(e,(0, 2, 1))
						g = K.exp(e - K.max(e, axis=-1, keepdims=True))
						
						if input_length>40:
							bound = -25
						elif input_length<35:
							bound = -10
						else:
							bound = -10
						k_max_e = K.T.set_subtensor(g[K.T.arange(g.shape[0]).dimshuffle(0,'x','x'),
													K.T.arange(g.shape[1]).dimshuffle('x',0,'x'), 
													K.T.argsort(g)[:,:,:bound]],
													0.0)
						
						s = K.sum(k_max_e,axis=-1,keepdims=True)
						sm_att = k_max_e / s
						return K.batch_dot(sm_att, mat)	
					return merge([attention, sentence], mode=_normalize_attention,
								  output_shape=(output_length, self.nr_hidden)) 
			class _max_self(object):#for self, output 1*hidden
				def __init__(self, nr_hidden):
					self.nr_hidden = nr_hidden
				def __call__(self, sentence, attention, input_length, output_length=1):
					def _normalize_attention(attmat):
						att = attmat[0]
						mat = attmat[1]
						e = K.max(att, axis=-1, keepdims=True)
						# 3d softmax
						e = K.permute_dimensions(e,(0, 2, 1))
						g = K.exp(e - K.max(e, axis=-1, keepdims=True))
						
						if input_length>40:
							bound = -25
						elif input_length<35:
							bound = -10
						else:
							bound = -10
						k_max_e = K.T.set_subtensor(g[K.T.arange(g.shape[0]).dimshuffle(0,'x','x'),
													K.T.arange(g.shape[1]).dimshuffle('x',0,'x'), 
													K.T.argsort(g)[:,:,:bound]],
													0.0)
						
						s = K.sum(k_max_e,axis=-1,keepdims=True)
						sm_att = k_max_e / s
						return K.batch_dot(sm_att, mat)	
					return merge([attention, sentence], mode=_normalize_attention,
								  output_shape=(output_length, self.nr_hidden)) 
			class _mean_self(object):#for self, output 1*hidden
				def __init__(self, nr_hidden):
					self.nr_hidden = nr_hidden
				def __call__(self, sentence, attention, input_length, output_length=1):
					def _normalize_attention(attmat):
						att = attmat[0]
						mat = attmat[1]
						e = K.mean(att, axis=-1, keepdims=True)
						# 3d softmax
						e = K.permute_dimensions(e,(0, 2, 1))
						g = K.exp(e - K.max(e, axis=-1, keepdims=True))
						
						if input_length>40:
							bound = -25
						elif input_length<35:
							bound = -10
						else:
							bound = -10
						k_max_e = K.T.set_subtensor(g[K.T.arange(g.shape[0]).dimshuffle(0,'x','x'),
													K.T.arange(g.shape[1]).dimshuffle('x',0,'x'), 
													K.T.argsort(g)[:,:,:bound]],
													0.0)
						
						s = K.sum(k_max_e,axis=-1,keepdims=True)
						sm_att = k_max_e / s
						return K.batch_dot(sm_att, mat)	
					return merge([attention, sentence], mode=_normalize_attention,
								  output_shape=(output_length, self.nr_hidden)) 
			class _Softmax(object):#for self,output len*hidden,no k_max/threshold,may need ques/ans_repear_vec
				def __init__(self, nr_hidden):
					self.nr_hidden = nr_hidden
				def __call__(self, embedding, attention, len):
					def _normalize_attention(attmat):
						att = K.permute_dimensions(attmat[0],(0, 2, 1))
						e = K.exp(att - K.max(att, axis=-1, keepdims=True))
						s = K.sum(e, axis=-1, keepdims=True)
						sm_att = e / s
						return attmat[1]*K.permute_dimensions(sm_att,(0, 2, 1))
					return merge([attention, embedding], mode=_normalize_attention,
								  output_shape=(len, self.nr_hidden))
			class _k_max_Softmax(object):#for self, output len*hidden, dont need mask
				def __init__(self, nr_hidden):
					self.nr_hidden = nr_hidden
				def __call__(self, embedding, attention, len):
					def _normalize_attention(attmat):
						att = K.permute_dimensions(attmat[0],(0, 2, 1))
						e = K.exp(att - K.max(att, axis=-1, keepdims=True))
						bound = len/2
						k_max_e = K.T.set_subtensor(e[K.T.arange(e.shape[0]).dimshuffle(0,'x','x'),
													K.T.arange(e.shape[1]).dimshuffle('x',0,'x'), 
													K.T.argsort(e)[:,:,:bound]],
													0.0)
						s = K.sum(e, axis=-1, keepdims=True)
						sm_att = e / s
						return attmat[1]*K.permute_dimensions(sm_att,(0, 2, 1))
					return merge([attention, embedding], mode=_normalize_attention,
								  output_shape=(len, self.nr_hidden))
			class _k_thread_Softmax(object):#for self, output len*hidden, dont need mask
				def __init__(self, nr_hidden):
					self.nr_hidden = nr_hidden
				def __call__(self, embedding, attention, len):
					def _normalize_attention(attmat):
						att = K.permute_dimensions(attmat[0],(0, 2, 1))
						e = K.exp(att - K.max(att, axis=-1, keepdims=True))
						s = K.sum(e, axis=-1, keepdims=True)
						sm_att = e / s
						threshold = 1.0/len
						k_threshold_e = T.switch(K.less(sm_att,threshold), 0.0, sm_att)
						new_s = T.clip(T.sum(k_threshold_e,axis=-1,keepdims=True),0.00001,1024)
						new_sm_att = k_threshold_e / new_s
						return attmat[1]*K.permute_dimensions(new_sm_att,(0, 2, 1))
					return merge([attention, embedding], mode=_normalize_attention,
								  output_shape=(len, self.nr_hidden))

			hidden_dim = model_param.hidden_dim
			question 	 = Input( shape=(model_param.enc_timesteps,), dtype='float32', name='question_base')
			question_len = Input( shape=(model_param.enc_timesteps,), dtype='float32', name='question_len')
			answer_len   = Input( shape=(model_param.dec_timesteps,), dtype='float32', name='answer_len')
			answer 		 = Input( shape=(model_param.dec_timesteps,), dtype='float32', name='answer_good_base')
			
			weights = np.load(embedding_file)
			weights[0] = np.zeros((weights.shape[1]))   

			QaEmbedding = Embedding(input_dim=weights.shape[0],#word num
									output_dim=weights.shape[1],#embedding size
									weights=[weights],
									# dropout=0.2,
									trainable=False)

			question_emb = QaEmbedding(question)
			answer_emb = QaEmbedding(answer)

			##=====mask filter
			ques_filter_repeat_len = RepeatVector(model_param.dec_timesteps)(question_len)
			ans_filter_repeat_len = RepeatVector(model_param.enc_timesteps)(answer_len)

			ques_repeat_ques = RepeatVector(model_param.enc_timesteps)(question_len)
			ans_repeat_ans = RepeatVector(model_param.dec_timesteps)(answer_len)
			
			ans_repeat_len = RepeatVector(model_param.hidden_dim)(answer_len)
			ans_repear_vec = Permute((2,1))(ans_repeat_len)

			ques_repeat_len = RepeatVector(model_param.hidden_dim)(question_len)
			ques_repear_vec = Permute((2,1))(ques_repeat_len)

			#======self output 1*hidden
			Softmax_self = _max_self(model_param.hidden_dim)
			Softmean_self = _mean_self(model_param.hidden_dim)
			###### self tanh ######
			tanh_q_D = Dense(model_param.hidden_dim,activation="tanh")
			tanh_q = TimeDistributed(tanh_q_D,name="tanh_q")
			tanh_a_D = Dense(model_param.hidden_dim,activation="tanh")
			tanh_a = TimeDistributed(tanh_a_D,name="tanh_a")
			temp_t_a = tanh_a(answer_emb)
			temp_t_q = tanh_q(question_emb)
			th_a_max = Softmax_self(answer_emb,temp_t_a,model_param.dec_timesteps)#batch*1*100
			th_q_max = Softmax_self(question_emb,temp_t_q,model_param.enc_timesteps)#batch*1*100
			th_a_mean = Softmean_self(answer_emb,temp_t_a,model_param.dec_timesteps)#batch*1*100
			th_q_mean = Softmean_self(question_emb,temp_t_q,model_param.enc_timesteps)#batch*1*100

			#======self output len*hidden
			Softmax = _k_max_Softmax(model_param.hidden_dim)
			###### self tanh ######
			tanh_q_D_full = Dense(model_param.hidden_dim,activation="tanh")
			tanh_q_full = TimeDistributed(tanh_q_D_full,name="tanh_q_full")
			tanh_a_D_full = Dense(model_param.hidden_dim,activation="tanh")
			tanh_a_full = TimeDistributed(tanh_a_D_full,name="tanh_a_full")
			temp_t_a_full = tanh_a_full(answer_emb)
			temp_t_q_full = tanh_q_full(question_emb)
			th_a_full = Softmax(answer_emb,temp_t_a_full,model_param.dec_timesteps)
			th_q_full = Softmax(question_emb,temp_t_q_full,model_param.enc_timesteps)

			##======intra and first co
			Attend = _Attention(model_param.enc_timesteps, model_param.dec_timesteps, hidden_dim)
			Attend_q = _Attention(model_param.enc_timesteps, model_param.enc_timesteps, hidden_dim)
			Attend_a = _Attention(model_param.dec_timesteps, model_param.dec_timesteps, hidden_dim)
			Align_mask = _SoftAlignment_mask(hidden_dim)
			Max_S = _max_soft(hidden_dim)
			Mean_S = _mean_soft(hidden_dim)	

			#co-att and intra-att
			ques_att = Attend(question_emb,answer_emb)
			ques_intra = Attend_q(question_emb,question_emb)
			ans_intra = Attend_a(answer_emb,answer_emb)
			ans_att = Attend(question_emb,answer_emb,reverse = True)
			print("ques_att._keras_shape",ques_att._keras_shape)
			print("ques_intra._keras_shape",ques_intra._keras_shape)
			print("ans_intra._keras_shape",ans_intra._keras_shape)

			#co Max and mean pooling, it is 1*hidden
			question_Max = Max_S(question_emb,ques_att,model_param.enc_timesteps)
			answer_Max = Max_S(answer_emb,ans_att,model_param.dec_timesteps)
			question_Mean = Mean_S(question_emb,ques_att,model_param.enc_timesteps)
			answer_Mean = Mean_S(answer_emb,ans_att,model_param.dec_timesteps)
			print answer_Max._keras_shape
			print answer_Mean._keras_shape
			
			#co alignment pooling, it is len*hidden
			answer_Soft = Align_mask(question_emb,ques_att,ques_filter_repeat_len,model_param.dec_timesteps)
			question_Soft = Align_mask(answer_emb,ans_att,ans_filter_repeat_len,model_param.enc_timesteps)
			print answer_Soft._keras_shape
			max_pool = Lambda(lambda x:K.max(x, axis=1,keepdims=True), output_shape=lambda x:(x[0], 1, x[2]))
			avg_pool = Lambda(lambda x:K.mean(x, axis=1,keepdims=True), output_shape=lambda x:(x[0], 1, x[2]))
			answer_Soft_max = max_pool(answer_Soft)
			answer_Soft_avg = avg_pool(answer_Soft)
			question_Soft_max = max_pool(question_Soft)
			question_Soft_avg = avg_pool(question_Soft)
			
			#intra Max and mean pooling,it is 1*hidden
			question_intra_Max = Max_S(question_emb,ques_intra,model_param.enc_timesteps)
			question_intra_Mean = Mean_S(question_emb,ques_intra,model_param.enc_timesteps)
			answer_intra_Max = Max_S(answer_emb,ans_intra,model_param.dec_timesteps)
			answer_intra_Mean = Mean_S(answer_emb,ans_intra,model_param.dec_timesteps)
			print("question_intra_Max._keras_shape",question_intra_Max._keras_shape)
			print("answer_intra_Max._keras_shape",answer_intra_Max._keras_shape)
			
			#intra alignment, it is len*hidden
			question_intra_Soft = Align_mask(question_emb,ques_intra,ques_repeat_ques,model_param.enc_timesteps)
			answer_intra_Soft = Align_mask(answer_emb,ans_intra,ans_repeat_ans,model_param.dec_timesteps)
			print answer_intra_Soft._keras_shape
			answer_intra_Soft_max = max_pool(answer_intra_Soft)
			answer_intra_Soft_avg = avg_pool(answer_intra_Soft)
			question_intra_Soft_max = max_pool(question_intra_Soft)
			question_intra_Soft_avg = avg_pool(question_intra_Soft)
			
			#concat emb with features
			question_emb = merge([question_emb,question_Soft_max,question_Soft_avg,question_Max,question_Mean,th_q_max,th_q_mean,question_intra_Max,question_intra_Mean,question_intra_Soft_max,question_intra_Soft_avg],mode='concat',concat_axis=1)
			answer_emb = merge([answer_emb,answer_Soft_max,answer_Soft_avg,answer_Max,answer_Mean,th_a_max,th_a_mean,answer_intra_Max,answer_intra_Mean,answer_intra_Soft_max,answer_intra_Soft_avg],mode='concat',concat_axis=1)
			print answer_emb._keras_shape

			##=====encoder
			SigmoidDense = Dense(hidden_dim,activation="sigmoid")
			TanhDense = Dense(hidden_dim,activation="tanh")

			QueTimeSigmoidDense = TimeDistributed(SigmoidDense,name="que_time_s")
			QueTimeTanhDense = TimeDistributed(TanhDense,name="que_time_t")

			AnsTimeSigmoidDense = TimeDistributed(SigmoidDense,name="ans_time_s")
			AnsTimeTanhDense = TimeDistributed(TanhDense,name="ans_time_t")

			question_sig = QueTimeSigmoidDense(question_emb)
			question_tanh = QueTimeTanhDense(question_emb)
			question_proj = merge([question_sig,question_tanh],mode="mul")

			answer_sig = AnsTimeSigmoidDense(answer_emb)
			answer_tanh = AnsTimeTanhDense(answer_emb)
			answer_proj = merge([answer_sig,answer_tanh],mode="mul")
			print answer_proj._keras_shape

			#=====second co att
			Attend2 = _Attention(model_param.enc_timesteps+10, model_param.dec_timesteps+10, hidden_dim)
			Align = _SoftAlignment(hidden_dim)
			ques_atten_metrics = Attend2(question_proj,answer_proj)
			ans_atten_metrics = Attend2(question_proj,answer_proj,reverse = True)
			print ques_atten_metrics._keras_shape
			Dense_Align_q = Dense(hidden_dim,activation="tanh")(question_proj)
			Dense_Align_a = Dense(hidden_dim,activation="tanh")(answer_proj)
			ques_atten_metrics_2 = Attend2(Dense_Align_q,answer_proj)
			ans_atten_metrics_2 = Attend2(question_proj,Dense_Align_a,reverse = True)
			answer_align = Align(question_proj,ques_atten_metrics,ques_atten_metrics_2,model_param.dec_timesteps+10)
			question_align = Align(answer_proj,ans_atten_metrics,ans_atten_metrics_2,model_param.enc_timesteps+10)
			print answer_align._keras_shape

			#========compare
			ans_sim_output = merge([answer_proj,answer_align],mode="mul")
			ques_sim_output = merge([question_proj,question_align],mode="mul")

			cnns = [Convolution1D(filter_length=filter_length,
							  nb_filter=hidden_dim,
							  activation='relu',
							  border_mode='same') for filter_length in [1,2,3,4,5]]

			cnn_feature = merge([cnn(ans_sim_output) for cnn in cnns], mode='concat')
			maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
			meanpool = Lambda(lambda x: K.mean(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
			cnn_maxpool = maxpool(cnn_feature)
			cnn_meanpool = meanpool(cnn_feature)

			OutputDense_max = Dense(hidden_dim,activation="tanh")
			OutputDense_mean = Dense(hidden_dim,activation="tanh")
			feature_max = OutputDense_max(cnn_maxpool)
			feature_mean = OutputDense_mean(cnn_meanpool)

			cnns1 = [Convolution1D(filter_length=filter_length,
							  nb_filter=hidden_dim,
							  activation='relu',
							  border_mode='same') for filter_length in [1,2,3,4,5]]

			cnn1_feature = merge([cnn(ques_sim_output) for cnn in cnns1], mode='concat')
			cnn1_maxpool = maxpool(cnn1_feature)
			cnn1_meanpool = meanpool(cnn1_feature)

			OutputDense1_max = Dense(hidden_dim,activation="tanh")
			OutputDense1_mean = Dense(hidden_dim,activation="tanh")
			feature1_max = OutputDense1_max(cnn1_maxpool)
			feature1_mean = OutputDense1_mean(cnn1_meanpool)

			feature_total = merge([feature_max,feature_mean,feature1_max,feature1_mean],mode='concat')
			FinalDense = Dense(hidden_dim,activation="tanh")
			feature_all = FinalDense(feature_total)

			Score_final = Dense(1)
			score = Score_final(feature_all)

			model = Model(input=[question,answer,question_len,answer_len],output=[score])
			return model
		#############################################
		#############################################
		#############################################
		question = Input(
			shape=(model_param.enc_timesteps,), dtype='float32', name='question_base')

		question_len = Input(shape=(model_param.enc_timesteps,), dtype='float32', name='question_len')

		good_answer = Input(
			shape=(model_param.dec_timesteps,), dtype='float32', name='answer_base')
		answers = Input(
			shape=(model_param.random_size,model_param.dec_timesteps,), dtype='float32', name='answer_bad_base')

		answers_length = Input(shape=(model_param.random_size,model_param.dec_timesteps,), dtype='float32', name='answers_length')
		good_answer_length = Input(shape=(model_param.dec_timesteps,),dtype='float32', name='good_answer_len')
		
		basic_model = get_core_model(model_param,embedding_file,vocab_size)

		good_similarity = basic_model([question, good_answer, question_len,good_answer_length])
		sim_list = []
		for i in range(model_param.random_size):
			convert_layer = Lambda(lambda x:x[:,i],output_shape=(model_param.dec_timesteps,))
			temp_tensor = convert_layer(answers)
			temp_length = convert_layer(answers_length)
			temp_sim = basic_model([question,temp_tensor,question_len,temp_length])
			sim_list.append(temp_sim)
		total_sim = merge(sim_list,mode="concat")
		total_prob = Lambda(lambda x: K.log(K.softmax(x)), output_shape = (model_param.random_size, ))(total_sim)
		

		prediction_model = Model(
			input=[question, good_answer,question_len,good_answer_length], output=good_similarity, name='prediction_model')
		prediction_model.compile(
			loss=lambda y_true, y_pred: y_pred, optimizer=Adam(lr=model_param.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
		training_model = Model(
			input=[question, answers,question_len,answers_length], output=total_prob, name='training_model')
		training_model.compile(
			loss=lambda y_true,y_pred: K.mean(y_true*(K.log(K.clip(y_true,0.00001,1)) - y_pred )) ,metrics=['accuracy'], optimizer=Adam(lr=model_param.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
		return training_model, prediction_model

	@staticmethod
	def get_model(model_param, embedding_file, vocab_size,model_type):
		if model_type == "listwise":
			return ModelFactory.get_listwise_model(model_param, embedding_file, vocab_size)