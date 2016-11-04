function [ cmatrix,accuracy ] = fivefoldval(trainimg,testimg,trainlabels,testlabels,validationimg)

	accuracy=0;
cmatrix=zeros(10);

for k=1:validationimg
	
	xi=[];
	for i=1:28
		xi=[xi img(i,:,k)];	%xi=kth row from inage matrix,convert xi to 784*1
	endfor	
	xi=double(xi)/255;
	
	t(1:10,1)=0.04;
	tk=trainlabels(k);
	t(tk+1,1)=0.96;	%index starts from 1 thats why tk+1
	
	netj=(xi*wji)';	%wj0=0,bias=0,net input on the node of hidden layer dimension-no of hidden nodes, wji ka transpose and then multiply it with xi => 15*1
	yj=1./(1+(exp(-netj)));

	netk=(yj'*wkj)';	%wk0=0,bias=0,net input on the node of output layer dimension-no of output nodes,10*1
	zk=1./(1+(exp(-netk)));	%10*1
	
	   [maxval,maxind]=max(zk(:));
	     cmatrix(tk+1,maxind)=cmatrix(tk+1,maxind)+1;
	      if tk == maxind-1
	        	accuracy++;
	      endif
endfor	
endfunction