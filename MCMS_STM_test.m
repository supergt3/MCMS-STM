function [predict_X,I_final,accuracy]=MCMS_STM_test(X,model,b,R,H,test_label,type)
%
if strcmp(type,'0')
phi=zeros(size(X,1),size(X,1));
for i=1:size(X,1)
    temp=[1:i-1,i+1:size(X,1)];
    temp2=kron([1:H],ones(1,(size(X,1)-1)/H));
    phi(i,temp)=temp2;
end
predict_X=zeros(size(X,1),size(X,2));
neig_mat=cell(1,size(X,2));
for k=1:size(X,2)
    neig_mat{k}=zeros(size(X,1),size(X,1));
    for i=1:size(X,1)
        for j=1:size(X,1)
            if i==j
                continue;
            end
            temp_y=0;
            for l=1:R  
            temp_y=temp_y+ttv(X{i,k},{model(i,phi(i,j)).Proj{:,l}},[1:ndims(X{i,k})]); 
            end
            temp_m=0;
            for l=1:R  
            temp_m=temp_m+ttv(X{j,k},{model(j,phi(j,i)).Proj{:,l}},[1:ndims(X{j,k})]); 
            end
            if temp_y+b(i,phi(i,j))>temp_m+b(j,phi(j,i))
                predict_X(i,k)=predict_X(i,k)+1;
                neig_mat{k}(i,j)=1;
            end
       end
    end
end

[B,I_final_temp]=max(predict_X);
I_final=zeros(1,size(X,2));
temp_B=zeros(size(predict_X));
temp_B(ones(size(X,1),1)*B==predict_X)=1;
for i=1:size(X,2)
    temp=neig_mat{i};
    temp2=temp_B(:,i);
    if sum(temp2)==1
        I_final(i)=I_final_temp(i);
    else
    temp(temp2==0,:)=0;
    temp(:,temp2==0)=0;
    [~,B_temp]=max(sum(temp,2));
    I_final(i)=B_temp;
    end
end
temp=I_final==test_label;
accuracy=size(find(temp(:)==1),1)/size(temp(:),1);
else
 phi=zeros(size(X,1),size(X,1));
for i=1:size(X,1)
    temp=[1:i-1,i+1:size(X,1)];
    temp2=kron([1:H],ones(1,(size(X,1)-1)/H));
    phi(i,temp)=temp2;
end
predict_X=zeros(size(X,1),size(X,2));
type_K{1}='Gauss';
type_K{2}='Polynomial';
CP_X=cell(size(X));
for i=1:size(X,1)
    for j=1:size(X,2)
        CP_X{i,j}=parafac_als(X{i,j},1);
    end
end
% predict_X_v=zeros(size(X,1),size(X,2));
c_num=size(X,1);
for k=1:size(X,2)
    for i=1:size(X,1)
        for j=1:size(X,1)
            if i==j
                continue;
            end
            temp_y=0;
            for r=1:R
            temp2=1;
            for l=1:ndims(X{1,1})
%                      if l==dims_temp
%                          continue;
%                      end
                    temp=0;
                    temp_v1=find(model(i,phi(i,j)).Proj{l,r}(:,1)~=0|model(i,phi(i,j)).Proj{l,r}(:,2)~=0);
                    for g1=1:size(temp_v1,1)
                        g=ceil(temp_v1(g1)/c_num);
                        m=mod(temp_v1(g1),c_num);
                        if m==0
                            m=c_num;
                        end
%                         if model(i,phi(i,j)).Proj{l,r}((g-1)*c_num+m,1)~=0
                            temp=temp+model(i,phi(i,j)).Proj{l,r}((g-1)*c_num+m,1)*kernel_F(CP_X{i,k}.U{l},model(1).CP_X{model(1).Label(g),g}.U{l},type_K{r});
%                         end
%                         if model(i,phi(i,j)).Proj{l,r}((g-1)*c_num+m,2)~=0
                            temp=temp+model(i,phi(i,j)).Proj{l,r}((g-1)*c_num+m,2)*kernel_F(CP_X{i,k}.U{l},model(1).CP_X{m,g}.U{l},type_K{r});
%                         end
                    end
                    temp2=temp2*temp;   
            end
            temp_y=temp_y+temp2;
            end
            temp_y=temp_y*CP_X{i,k}.lambda;
            temp_m=0;
            for r=1:R
            temp2=1;
            for l=1:ndims(X{1,1})
%                      if l==dims_temp
%                          continue;
%                      end
                    temp=0;
                    temp_v1=find(model(j,phi(j,i)).Proj{l,r}(:,1)~=0|model(j,phi(j,i)).Proj{l,r}(:,2)~=0);
                    for g1=1:size(temp_v1,1)
                        g=ceil(temp_v1(g1)/c_num);
                        m=mod(temp_v1(g1),c_num);
                        if m==0
                            m=c_num;
                        end
%                         if model(j,phi(j,i)).Proj{l,r}((g-1)*c_num+m,1)~=0
                            temp=temp+model(j,phi(j,i)).Proj{l,r}((g-1)*c_num+m,1)*kernel_F(CP_X{j,k}.U{l},model(1).CP_X{model(1).Label(g),g}.U{l},type_K{r});
%                         end
%                         if model(j,phi(j,i)).Proj{l,r}((g-1)*c_num+m,2)~=0
                            temp=temp+model(j,phi(j,i)).Proj{l,r}((g-1)*c_num+m,2)*kernel_F(CP_X{j,k}.U{l},model(1).CP_X{m,g}.U{l},type_K{r});
%                         end
                    end
                    temp2=temp2*temp;   
            end
             temp_m=temp_m+temp2;
            end
            temp_m=temp_m*CP_X{j,k}.lambda;
            if temp_y+b(i,phi(i,j))>temp_m+b(j,phi(j,i))
                predict_X(i,k)=predict_X(i,k)+1;
            end
%             predict_X_v(i,k)=temp_y+b(i,phi(i,j));
       end
    end
end
[~,I_final]=max(predict_X);
temp=I_final==test_label;
accuracy=size(find(temp(:)==1),1)/size(temp(:),1);
end
end