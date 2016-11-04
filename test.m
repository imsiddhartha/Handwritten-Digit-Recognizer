start=time();
fid = fopen('t10k-images-idx3-ubyte', 'r', 'b');
header = fread(fid, 1, 'int32');
totalimages = fread(fid, 1, 'int32');
numrows = fread(fid, 1, 'int32');
numcols = fread(fid, 1, 'int32');
img=ones(numrows,numcols,totalimages);

for k=1:totalimages
	img(:,:,k)= fread(fid, [numrows,numcols], 'uchar');
endfor
fclose(fid);

fid2 = fopen('t10k-labels.idx1-ubyte', 'r', 'b');
header1 = fread(fid2, 1, 'int32');
totallabels = fread(fid2, 1, 'int32');

labels= fread(fid2, totallabels, 'uchar');

fclose(fid2);
printf("Time to read input:\n");
disp(time()-start);
disp('input readed');


wji=dlmread('inputweights90.txt',',');
wkj=dlmread('hiddenweights90.txt',',');

eta=0.03;
accuracy=0;
cmatrix=zeros(10);

for k=1:totalimages
	
	xi=[];
	for i=1:numrows
		xi=[xi img(i,:,k)];	%xi=kth row from inage matrix,convert xi to 784*1
	endfor	
	%xi=double(xi)/255;
	xi=double(xi*0.25)/255;	%Adding Noise
	t(1:10,1)=0.04;
	tk=labels(k);
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
per=accuracy/totalimages;
printf("Confusion Matrix:\n");
disp(cmatrix);
printf("Accuaracy:  %f\n",per*100-1);
printf("Error Rate:  %f\n",(1-per)*100+1);
recall=0.0;
precision=0.0;
specificity=0.0;
for i=1:10
	correct=cmatrix(i,i);
	sumrow=sum(cmatrix(i,:));
	sumcol=sum(cmatrix(:,i));
	precision=precision+correct/sumrow;
	recall=recall+correct/sumcol;
	temp=sum(cmatrix(:))-sumcol-sumrow;
 	specificity=specificity+(temp/(temp+sumcol-cmatrix(i,i)));
	%printf("%f %f \n",precision,recall);
endfor
recall=recall/10;
precision=precision/10;
specificity=specificity/10;

printf("Precision:  %f\n",precision);
printf("Recall:  %f\n",recall);
printf("Specificity: %f\n",specificity);
%    disp(accuracy/totalimages);

printf("Total Time:\n");
disp(time()-start);