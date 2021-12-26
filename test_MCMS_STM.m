addpath(genpath(['..\H_STM_Experiment']));
addpath(genpath(['D:\Documents\MATLAB\tensor_toolbox']));
addpath(genpath(['..\STM']));
addpath(genpath(['d:\Documents\MATLAB\libsvm-3.14']));

str='D:/samples/airplane_slice/test_time/B52_scale/';
samples(1).X=heteroFeature_construct(str,1);
str='D:/samples/airplane_slice/test_time/C130_scale/';
samples(2).X=heteroFeature_construct(str,1);
str='D:/samples/airplane_slice/test_time/B1_scale/';
samples(3).X=heteroFeature_construct(str,1);
str='D:/samples/airplane_slice/test_time/C17_scale/';
samples(4).X=heteroFeature_construct(str,1);
str='D:/samples/airplane_slice/test_time/A10_scale/';
samples(5).X=heteroFeature_construct(str,1);
samples_num=zeros(1,size(samples,2));
for i=1:size(samples,2)
    samples_num(i)=size(samples(i).X,2);
end
scale_num=[[120,120];[70,70];[80,80];[80,80];[100,100];];
%%
%------------------------------MCMS-STM-----------------------------------%
num_v=1;
accuracy=zeros(15,num_v);   
test_num=zeros(1,num_v);
R=2;
H=1;
C=10;
accuracy_c=zeros(1,5);
for i=2:num_v
    samples_class_train=[];
    Label=[];
    for k=1:size(samples,2)
        samples_class_train=[samples_class_train,samples(k).X];
        Label=[Label,k*ones(1,samples_num(k))];
    end
    samples_class_train2=[];
     for k=1:size(samples,2)
        samples_class_train2=[samples_class_train2;samples_class_train];
     end
    [model,b]=MCMS_STM_train(samples_class_train2,Label,C,R,H,'0');
    test_X=[];
    test_label=[];
    for k=1:size(samples,2)
        test_X=[test_X,samples(k).X(:,[round(samples_num(k)*(i-1)/num_v)+1:round(samples_num(k)*i/num_v)])];
        test_label=[test_label,k*ones(size([round(samples_num(k)*(i-1)/num_v)+1:round(samples_num(k)*i/num_v)]))];
    end
    test_X2=[];
     for k=1:size(samples,2)
        test_X2=[test_X2;test_X];
     end
    [predict_X,predict_c,accuracy(i)]=MCMS_STM_test(test_X2,model,b,R,H,test_label,'0');
    temp=predict_c==test_label;
    for count=1:5
        accuracy_c(count)=accuracy_c(count)+sum(temp(test_label==count));
    end
    test_num(i)=size(test_label,2);
end
acc_MCMS_STM=(accuracy*test_num')/sum(test_num);
acc_MCMS_STM_c=accuracy_c./(ones(15,1)*samples_num);


