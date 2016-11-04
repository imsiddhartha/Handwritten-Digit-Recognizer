start=time();
fid = fopen('train-images.idx3-ubyte', 'r', 'b');
header = fread(fid, 1, 'int32')
totalimages = fread(fid, 1, 'int32')
numrows = fread(fid, 1, 'int32')
numcols = fread(fid, 1, 'int32')

img=ones(numrows,numcols,totalimages);
for k=1:totalimages
	%disp(k);
	img(:,:,k)= fread(fid, [numrows,numcols], 'uchar');
endfor

fclose(fid);
%for k=1:totalimages
%	disp("image:");
%	for i=1:numrows
%		for j=1:numcols
%			printf("%d ",img(k,i,j));
%		endfor
%	printf("\n");
%	endfor	
%	if k==2
%		break;
%	endif	
%endfor

%disp(totalimages);
fid2 = fopen('train-labels.idx1-ubyte', 'r', 'b');
header1 = fread(fid2, 1, 'int32')
totallabels = fread(fid2, 1, 'int32')

labels= fread(fid2, totallabels, 'uchar');

fclose(fid2);
disp('input readed');
disp(time()-start);
%for i=1:totallabels
%	header1 = fread(fid2, 1, 'uchar');
%	labels(i)=header1;
%endfor



hiddenlayers=90;
wji=rand(numcols*numrows,hiddenlayers)*0.09; 	%num of hidden nodes is 15	bias is 0th row
wkj=rand(hiddenlayers,10)*.09;
eta=0.5;
threshold=1;
it=0;
J=102;
format long
while(J>100)
J=0;		
for k=1:totalimages
	
	xi=[];
	for i=1:numrows
		xi=[xi img(i,:,k)];	%xi=kth row from inage matrix,convert xi to 784*1
	endfor	
	xi=double(xi)/255;
	
	t(1:10,1)=0.02;
	tk=labels(k);
	t(tk+1,1)=0.95;	%index starts from 1 thats why tk+1
	
	netj=(xi*wji)';	%wj0=0,bias=0,net input on the node of hidden layer dimension-no of hidden nodes, wji ka transpose and then multiply it with xi => 15*1
	yj=1./(1+(exp(-netj)));

	netk=(yj'*wkj)';	%wk0=0,bias=0,net input on the node of output layer dimension-no of output nodes,10*1
	zk=1./(1+(exp(-netk)));	%10*1
	
	dfk=zk.*(1-zk);	%10*1
	dfj=yj.*(1-yj);	%15*1

	dk=(t-zk).*dfk;	%10*1	
	dwkj=eta*yj*dk';
	wkj=wkj+dwkj;
	
	dj=dfj.*(wkj*dk);	%summation aayega! 10*1
	dwji=eta*xi'*dj';
	wji=wji+dwji;

	%wkj=wkj+dwkj- (0.02*eta*wkj);	%weight Decay!
	%wji=wji+dwji-(0.02*eta*wji);
	 J=J+1/2*sum((t-zk).^2);
	 % J=J+1/2*sum((t-zk).^2);
endfor	
%disp(it);
disp(J);
 if mod(it,1000) == 0
    	disp(it);
    	disp(J);
    	dlmwrite('ipweights.txt',wji);
	dlmwrite('hiddenweights.txt',wkj);
endif
it=it+1;
endwhile
dlmwrite('inputweights90.txt',wji);
dlmwrite('hiddenweights90.txt',wkj);
disp(it);
disp(time()-start);


validationimg=totalimages/5;
testimg=ones(numrows,numcols,validationimg);
for i=1:validationimg
	testlabels(i,:)=labels(i,:);
	testimg(:,:,i)=img(:,:,i);
endfor
w22=size(testlabels);
w222=size(testimg);
trainimg=ones(numrows,numcols,totalimages-validationimg);

for i=validationimg+1:totalimages-validationimg
	trainlabels(i,:)=labels(i,:);
	trainimg(:,:,i)=img(:,:,i);
endfor
answer=zeros;
for j=1:5
	[confusionmat accuracy]=fivefoldval(trainimg,testimg,trainlabels,testlabels,validationimg);
	answer=answer+confusionmat;
endfor
per=accuracy/(validationimg*5);
printf("First fold\n");
printf("Accuaracy:  %f\n",per*100-1);
printf("Error Rate:  %f\n",(1-per)*100);
printf("Precision:  %f\n",per);
printf("Recall:  %f\n",per);
disp(confusionmat);
%2nd

for i=1+validationimg:2*validationimg
	testlabels(i,:)=labels(i,:);
	testimg(:,:,i)=img(:,:,i);
endfor
w22=size(testlabels);
w222=size(testimg);
trainimg=ones(numrows,numcols,totalimages-validationimg);

for i=1:validationimg
	trainlabels(i,:)=labels(i,:);
	trainiimg(:,:,i)=img(:,:,i);
endfor
for i=2*validationimg+1:totalimages
	trainlabels(i,:)=labels(i,:);
	trainiimg(:,:,i)=img(:,:,i);
endfor

answer=zeros;
for j=1:5
	[confusionmat accuracy]=fivefoldval(trainimg,testimg,trainlabels,testlabels,validationimg);
	answer=answer+confusionmat;
endfor

per=accuracy/(validationimg*5);
printf("Second fold\n");
printf("Accuaracy:  %f\n",per*100-1);
printf("Error Rate:  %f\n",(1-per)*100);
printf("Precision:  %f\n",per);
printf("Recall:  %f\n",per);
disp(confusionmat);
%3rd

for i=validationimg*2+1:3*validationimg
	testlabels(i,:)=labels(i,:);
	testimg(:,:,i)=img(:,:,i);
endfor
w22=size(testlabels);
w222=size(testimg);
trainimg=ones(numrows,numcols,totalimages-validationimg);

for i=1:2*validationimg
	trainlabels(i,:)=labels(i,:);
	trainiimg(:,:,i)=img(:,:,i);
endfor
for i=3*validationimg+1:totalimages
	trainlabels(i,:)=labels(i,:);
	trainiimg(:,:,i)=img(:,:,i);
endfor

answer=zeros;
for j=1:5
	[confusionmat accuracy]=fivefoldval(trainimg,testimg,trainlabels,testlabels,validationimg);
	answer=answer+confusionmat;
endfor
per=accuracy/(validationimg*5);
printf("Third fold\n");
printf("Accuaracy:  %f\n",per*100-1);
printf("Error Rate:  %f\n",(1-per)*100);
printf("Precision:  %f\n",per);
printf("Recall:  %f\n",per);
disp(confusionmat);
%4th

for i=validationimg*3+1:4*validationimg
	testlabels(i,:)=labels(i,:);
	testimg(:,:,i)=img(:,:,i);
endfor
trainimg=ones(numrows,numcols,totalimages-validationimg);

for i=1:3*validationimg
	trainlabels(i,:)=labels(i,:);
	trainiimg(:,:,i)=img(:,:,i);
endfor
for i=4*validationimg+1:totalimages
	trainlabels(i,:)=labels(i,:);
	trainiimg(:,:,i)=img(:,:,i);
endfor

answer=zeros;
for j=1:5
	[confusionmat accuracy]=fivefoldval(trainimg,testimg,trainlabels,testlabels,validationimg);
	answer=answer+confusionmat;
endfor
per=accuracy/(validationimg*5);
printf("Fourth fold\n");
printf("Accuaracy:  %f\n",per*100-1);
printf("Error Rate:  %f\n",(1-per)*100);
printf("Precision:  %f\n",per);
printf("Recall:  %f\n",per);
disp(confusionmat);
%5th

for i=validationimg*4+1:5*validationimg
	testlabels(i,:)=labels(i,:);
	testimg(:,:,i)=img(:,:,i);
endfor

trainimg=ones(numrows,numcols,totalimages-validationimg);

for i=1:4*validationimg
	trainlabels(i,:)=labels(i,:);
	trainiimg(:,:,i)=img(:,:,i);
endfor

answer=zeros;
for j=1:5
	[confusionmat accuracy]=fivefoldval(trainimg,testimg,trainlabels,testlabels,validationimg);
	answer=answer+confusionmat;
endfor
per=accuracy/(validationimg*5);
printf("Fifth fold\n");
printf("Accuaracy:  %f\n",per*100-1);
printf("Error Rate:  %f\n",(1-per)*100);
printf("Precision:  %f\n",per);
printf("Recall:  %f\n",per);
disp(confusionmat);