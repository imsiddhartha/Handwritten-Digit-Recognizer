start=time();
fid = fopen('t10k-images-idx3-ubyte', 'r', 'b');
header = fread(fid, 1, 'int32');
totalimages = fread(fid, 1, 'int32')
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

wji=dlmread('inputweights90.txt',',');
wkj=dlmread('hiddenweights90.txt',',');

accuracy=0;
cmatrix=zeros(10);

x(1,1)=100.0;x(1,2)=91.86;
x(2,1)=200.0;x(2,2)=87.49;
x(3,1)=300.0;x(3,2)=73.04;
x(4,1)=400.0;x(4,2)=45.84;
x(5,1)=500.0;x(5,2)=11.02;
x(6,1)=600.0;x(6,2)=4.73;
x(7,1)=700.0;x(7,2)=3.62;
%disp(x);
figure;
plot(x(1:7,1),x(1:7,2),'pr')
xlabel('No. of Features')
ylabel('Error Rate')
title('Error-Rate vs No .of features')
line(x(1:7,1),x(1:7,2))
hold;
axis([0,700,0,100])

%legend('w1','Error-Rates')

%for k=1:totalimages
	
%	xi=[];
%	for i=1:numrows
%		xi=[xi img(i,:,k)];	%xi=kth row from inage matrix,convert xi to 1*784
%	endfor	
	%www=size(xi)

%	xi=double(xi)/255;
	
	%xi=xi(:,1:100);
	%xi=xi(:,1:200);
	%www=size(xi)
	%wji=wji(1:100,:);
	%wji=wji(1:200,:);
	%www=size(wji)	
%	t(1:10,1)=0.04;
%	tk=labels(k);
%	t(tk+1,1)=0.96;	%index starts from 1 thats why tk+1
	
%	netj=(xi*wji)';	%wj0=0,bias=0,net input on the node of hidden layer dimension-no of hidden nodes, wji ka transpose and then multiply it with xi => 15*1
%	yj=1./(1+(exp(-netj)));

%	netk=(yj'*wkj)';	%wk0=0,bias=0,net input on the node of output layer dimension-no of output nodes,10*1
%	zk=1./(1+(exp(-netk)));	%10*1
%	   [maxval,maxind]=max(zk(:));
%	      if tk == maxind-1
%	        	accuracy++;
%	      endif
%endfor
%disp(accuracy);
%per=accuracy/totalimages;
%printf("Accuaracy:  %f\n",per*100-1);
%printf("Error Rate:  %f\n",(1-per)*100+1);
