start=time();
fid = fopen('train-images.idx3-ubyte', 'r', 'b');
header = fread(fid, 1, 'int32');
totaltrainimages = fread(fid, 1, 'int32');
numrows = fread(fid, 1, 'int32');
numcols = fread(fid, 1, 'int32');

trainimg=ones(numrows,numcols,totaltrainimages);
for k=1:totaltrainimages
	%disp(k);
	trainimg(:,:,k)= fread(fid, [numrows,numcols], 'uchar');
endfor

fclose(fid);

fid2 = fopen('train-labels.idx1-ubyte', 'r', 'b');
header1 = fread(fid2, 1, 'int32');
totaltrianlabels = fread(fid2, 1, 'int32');
trainlabels= fread(fid2, totaltrianlabels, 'uchar');
fclose(fid2);
disp('input readed');


fid = fopen('t10k-images-idx3-ubyte', 'r', 'b');
header = fread(fid, 1, 'int32');
totaltestimages = fread(fid, 1, 'int32');
numrows = fread(fid, 1, 'int32');
numcols = fread(fid, 1, 'int32');
testimg=ones(numrows,numcols,totaltestimages);

for k=1:totaltestimages
	testimg(:,:,k)= fread(fid, [numrows,numcols], 'uchar');
endfor
fclose(fid);

fid2 = fopen('t10k-labels.idx1-ubyte', 'r', 'b');
header1 = fread(fid2, 1, 'int32');
totaltestlabels = fread(fid2, 1, 'int32');

testlabels= fread(fid2, totaltestlabels, 'uchar');
fclose(fid2);
accuracy=0;
for k=1:totaltestimages
	disp(k);
	xi=[];
	for i=1:numrows
		xi=[xi testimg(i,:,k)];	%xi=kth row from inage matrix,convert xi to 784*1
	endfor	
	dist=zeros;
	for j=1:totaltrainimages
		yi=[];
		for i=1:numrows
			yi=[yi trainimg(i,:,j)];	%xi=kth row from inage matrix,convert xi to 784*1
		endfor
		temp=xi-yi;
		temp=temp.^2;
		temp1=sum(temp(:));
		temp1=sqrt(temp1);
		dist(j)=temp1;
	endfor
	[val,ind]=min(dist(:));
	if testlabels(k) == trainlabels(ind)
		accuracy++;
	end
	%if k==20
	%	break;
	%endif		
endfor
disp(time()-start);
per=accuracy/totaltestimages;
printf("Accuaracy:  %f\n",per*100);
printf("Error Rate:  %f\n",(1-per)*100);
printf("Precision:  %f\n",per);
printf("Recall:  %f\n",per);
