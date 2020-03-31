import os
import sys
import numpy as np
import getopt
import tensorflow.keras.models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import multiprocessing
from multiprocessing import Pool, Manager

cores=int((multiprocessing.cpu_count())/2)

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

path=('./models/')
model_path=os.listdir('./models')

t = Tokenizer(filters='?\t')
AA=['s', 'g', 't', 'l', 'v', 'a', 'k', 'r', 'w', 'e', 'p', 'y', 'd', 'i', 'f', 'q', 'n', 'm', 'c', 'h']

AA_dic = { "CYS":"C","SER":"S","THR":"T","PRO":"P","ALA":"A", "GLY":"G","ASN":"N","ASP":"D","GLU":"E","GLN":"Q", "HIS":"H","ARG":"R","LYS":"K","MET":"M","ILE":"I", "LEU":"L","VAL":"V","PHE":"F","TYR":"Y","TRP":"W"}
al=' ABCDEFGHIJKLMNOPQRSTUVWXYZ'

#AA=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
cdr_dict={'H1': [9,4,14],'H2':[7,12,27],'H3':[7,3,27],'L1':[7,7,21],'L2':[7,5,14],'L3':[7,4,20]}

t.fit_on_texts(AA)

mode=None
chain={}
all_list=[]
seq=[]
output=None
auto_out={}
CPU=None

def help():
  message=open('README').read()
  print(message)
  return
 
def main():
  try:
    opts, args = getopt.getopt(sys.argv[1:],"H:L:i:o:n:",["fasta","seq","pdb","help"])
  except getopt.GetoptError as err:
    print str(err)
    help()
    sys.exit(1)
  global mode, chain, seq, output, all_list, auto_out, CPU
  for opt,arg in opts:
    if ( opt == "--seq" ):
	mode = "seq" 
    elif ( mode == "seq" ) and ( opt == "-H"):
	if arg[-5:] == ".list":
		chain["H"]=open(arg).readlines()
		auto_out["H"]=arg
	else:
		chain["H"]=[arg]
    elif ( mode == "seq" ) and ( opt == "-L"):
	if arg[-5:] == ".list":
		chain["L"]=open(arg).readlines()
		auto_out["L"]=arg
	else:
		chain["L"]=[arg]
    if ( opt == "--fasta" ):
    	mode = "fasta"
    elif ( mode == "fasta" ) and ( opt == "-i"):
	if arg[-5:] == ".list":
		seq = open(arg).readlines()
		auto_out["fasta"] = arg
	else:
		seq = [arg]
    if ( opt == "--pdb"):
	mode = "pdb"
    elif ( mode == "pdb") and ( opt == "-H"):
        if arg[-5:] == ".list":
                chain["H"]=open(arg).readlines()
		auto_out["H"]=arg
        else:
                chain["H"]=[arg]
    elif ( mode == "pdb") and ( opt == "-L"):
        if arg[-5:] == ".list":
                chain["L"]=open(arg).readlines()
		auto_out["L"]=arg
        else:
                chain["L"]=[arg]
    if ( opt == "-o"):
	output = arg
    if ( opt == "-n"):
	CPU=arg
    elif ( opt == "-h") or ( opt == "--help"):
      help()
      sys.exit(1)
    


if __name__ == '__main__':
	main()


if CPU == None:
	num_cpus = cores
else:
	num_cpus=int(CPU)

pp=0

ls={0.2:'AAA'}
ls=Manager().dict()

def model(val):
  global pp, new_model, path, cdr
  if pp==0:
	new_model = tensorflow.keras.models.load_model(path+'model_'+cdr+'.h5')
	pp+=1
	result=new_model.predict(val)
	ls[result[0][0]]=' '.join(t.sequences_to_texts(val)).upper()
	#print(' '.join(t.sequences_to_texts(val)).upper()+' '+str(result[0][0]))
  else:
	result=new_model.predict(val)
	ls[result[0][0]]=' '.join(t.sequences_to_texts(val)).upper()
	#print(' '.join(t.sequences_to_texts(val)).upper()+' '+str(result[0][0]))


def par(current_word):
  global ls, cdr_dict
  seq_list=[]
  current_word=current_word.replace('',' ').strip()
  encoded = t.texts_to_sequences([current_word])[0]
  pat_len=cdr_dict[cdr][0]
  ran=range(cdr_dict[cdr][1],cdr_dict[cdr][2]+1)
  for n in ran:
  	for a in range(0,len(current_word.replace(' ',''))-pat_len*2-n+1):
                val= pad_sequences([encoded[a:a+pat_len]+encoded[a+pat_len+n:a+pat_len*2+n]], maxlen=pat_len*2, padding='pre')
                val=np.array(val)
		seq_list.append(val)
  if __name__ == '__main__':
	pool = multiprocessing.Pool(processes=num_cpus)
	pool.map(model,seq_list)
	pool.close()
	pool.join()
  
  #for line in seq_list:
	#(line)
  sentence=(current_word.replace(' ',''))
  dic=dict(ls)
  ls.clear()
  point=(str(dic[max(dic)]).replace(' ',''))
  fp=point[:pat_len]
  rp=point[pat_len:]
  CDR=sentence[sentence.find(fp)+pat_len:sentence.find(rp)]
  if max(dic)!=0:
    return(CDR)
    #return(dic)
  else:
    return('None')
  dic.clear()

	
chainlist=['H1','H2','H3','L1','L2','L3']

result_heavy=[]
result_light=[]
result_final=[]


def fasta_file(seq_file):
	fasta_dic={}
	lines=open(seq_file).read().split('>')[1:]
	for line in lines:
		line=line.strip().split('\n')
		head=line[0]
		body=''.join(line[1:])
		fasta_dic[head]=body
	return fasta_dic

def pdb_file(pdb_file):
	global AA_dic, al
	pdb_dic={}
	MODRES=[]
	pdb = pdb_file.split(':')[0]
	chain_info = pdb_file.split(':')[1]
	lines=open(pdb).readlines()
	flag1=0
	seq=''
	for pdb_line in lines:
		if pdb_line[0:6]=='MODRES':
			MODRES.append(pdb_line)	
		elif (pdb_line[0:4]=='ATOM' or pdb_line[0:6]=='HETATM') and pdb_line[17:20]!='HOH' and pdb_line[21:22] == chain_info and ((float(pdb_line[22:26])+float(al.find(pdb_line[26]))/100)>float(flag1)):
			if pdb_line[17:20] not in AA_dic:
				for MOD_line in MODRES:
					if MOD_line[18:23] == pdb_line[22:27] and MOD_line[16:17] == pdb_line[21:22] and MOD_line[12:15] == pdb_line[17:20]:
						if MOD_line[24:27] in AA_dic:
							seq+=(AA_dic[MOD_line[24:27]])
							flag1=float(pdb_line[22:26])+float(al.find(pdb_line[26]))/100

			else:
				flag1=float(pdb_line[22:26])+float(al.find(pdb_line[26]))/100
				seq+=(AA_dic[pdb_line[17:20]])

	
	return seq

result_heavy2=[]
result_light2=[]

if mode == "seq":
	if chain != {}:
		for chainline in chain:
			if chainline == "H":
				for num,n in enumerate(chain[chainline]):
					n=n.strip()
					for cdr in chainlist[:3]:
						ans=(par(n))
						result_heavy.append(ans)
					
					print(str(num+1)+'_CDR_heavy'+'\t'+'\t'.join(result_heavy[-3:]))
					result_final.append(str(num+1)+'_CDR_heavy'+'\t'+'\t'.join(result_heavy[-3:]))
					result_heavy2.append(str(num+1)+'_CDR_heavy'+'\t'+'\t'.join(result_heavy[-3:]))
					
			elif chainline == "L":
				for num,n in enumerate(chain[chainline]):
					n=n.strip()
					for cdr in chainlist[3:]:
               					ans=(par(n))
						result_light.append(ans)
					print(str(num+1)+'_CDR_light'+'\t'+'\t'.join(result_light[-3:]))
					result_final.append(str(num+1)+'_CDR_light'+'\t'+'\t'.join(result_light[-3:]))
					result_light2.append(str(num+1)+'_CDR_light'+'\t'+'\t'.join(result_light[-3:]))
		if output != None:
			open(output,'w').write('\n'.join(result_final))
		elif output == None and auto_out != {}:
			for key in auto_out:
				if key == "H":
					open(auto_out[key]+'_H.out','w').write('\n'.join(result_heavy2))
				elif key == "L":
					open(auto_out[key]+'_L.out','w').write('\n'.join(result_light2))
					
	else:
		help()
		sys.exit(1)

elif mode == "fasta":
    	if seq != []:
		for n in seq:
			n=n.strip()
			seq_dic=fasta_file(n)
			for seq_chain in seq_dic:
				if seq_chain=="heavy":
					for cdr in chainlist[:3]:
						ans=(par(seq_dic[seq_chain]))
						result_heavy.append(ans)
					print(n.replace('.fasta','')+'_heavy'+'\t'+'\t'.join(result_heavy[-3:]))
					result_final.append(n.replace('.fasta','')+'_heavy'+'\t'+'\t'.join(result_heavy[-3:]))
				elif seq_chain=="light":
		        		for cdr in chainlist[3:]:
	        	        		ans=(par(seq_dic[seq_chain]))
	                        		result_light.append(ans)
					print(n.replace('.fasta','')+'_light'+'\t'+'\t'.join(result_light[-3:]))
					result_final.append(n.replace('.fasta','')+'_light'+'\t'+'\t'.join(result_light[-3:]))

		if output != None:    	
			open(output,'w').write('\n'.join(result_final))
		elif output == None and auto_out != {}:
			open(auto_out["fasta"]+".out",'w').write('\n'.join(result_final))
    	else:
    		help()
		sys.exit(1)

elif mode == "pdb":
	if chain != {}:
		for chainline in chain:
			if chainline == "H":
				for n in chain[chainline]:
					n=n.strip()	
					for cdr in chainlist[:3]:
						ans=(par(pdb_file(n)))
						result_heavy.append(ans)
					chain_info = n.split(':')[1]
					print(n.replace('.pdb','')+'_heavy'+'\t'+'\t'.join(result_heavy[-3:]))
					result_final.append(n.replace('.pdb','')+'_heavy'+'\t'+'\t'.join(result_heavy[-3:]))
					result_heavy2.append(n.replace('.pdb','')+'_heavy'+'\t'+'\t'.join(result_heavy[-3:]))
                        elif chainline == "L":
				for n in chain[chainline]:
					n=n.strip()
                                	for cdr in chainlist[3:]:
                                        	ans=(par(pdb_file(n)))
                                        	result_light.append(ans)
					chain_info = n.split(':')[1]
					print(n.replace('.pdb','')+'_light'+'\t'+'\t'.join(result_light[-3:]))
					result_final.append(n.replace('.pdb','')+'_light'+'\t'+'\t'.join(result_light[-3:]))
					result_light2.append(n.replace('.pdb','')+'_light'+'\t'+'\t'.join(result_light[-3:]))
		if output != None:
			open(output,'w').write('\n'.join(result_final))
                elif output == None and auto_out != {}:
                        for key in auto_out:
                                if key == "H":
                                        open(auto_out[key]+'_H.out','w').write('\n'.join(result_heavy2))
                                elif key == "L":
                                        open(auto_out[key]+'_L.out','w').write('\n'.join(result_light2))

	else:
		help()
		sys.exit(1)

elif mode == None:
	help()
	sys.exit(1)

