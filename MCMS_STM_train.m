function [model,b]=MCMS_STM_train(X,Label,C,R,H,type)
% MCMS_STM_train: training the MCMS-STM
%
% USAGE: [model,b]=MCMS_STM_train(X,Label,C,R,H,type)
%
% INPUT:
%
%   X: M x N x p cell loading with training samples tensor. M and N denote the
%   number of classes and the number of samples, respectively.
%   y: n x 1 vector of labels, 1 to M
%   C: a regularization parameter such that 0 <= alpha_i <= C
%   R: the CP rank of projection tensor
%   H: H=1 indicates the OVR version of MCMS-STM and H=M-1 indicates the OVO
%   version of MCMS-STM.
%   type: '0' denotes the linear version of MCMS-STM
%
% OUTPUT:
%
%   moder: parameters of MCMS-STM
%   b: scalar bias (offset) term in MCMS-STM
%   writtten by Tong Gao, 2021
% addpath(genpath(['d:\Documents\MATLAB\tensor_toolbox'])); 
addpath(genpath(['..\tensor_toolbox']));
if strcmp(type,'0') %linear version
tol2=0.01;
length=size(X,2);
%initialize the projection vectors
for i=1:size(X,1)
    for l=1:R
        for j=1:ndims(X{i,1})
            for h=1:H
                model(i,h).Proj{j,l}=rand(size(X{i,1},j),1);
            end
        end
    end
end
%recording maping relation for 1 versus m
phi=zeros(size(X,1),size(X,1));
for i=1:size(X,1)
    temp=[1:i-1,i+1:size(X,1)];
    temp2=kron([1:H],ones(1,(size(X,1)-1)/H));
    phi(i,temp)=temp2;
end
%recording the number of orders for different tensors
iterNum=1;
tensorDims=ndims(X{1,1});
alpha_num=size(X,1)*size(X,2);
c_num=size(X,1);
 %begin iterating
%  Iter_erroT=0;
while true
    model_temp=model;
    ntensorDim=zeros(size(X,1),1);
    for j=1:size(X,1)
        ntensorDim(j)=size(X{j,1},mod(iterNum,tensorDims)+1);
    end
    %calculating the coefficients of projection vectors
    wCoeff=zeros(size(X,1),R,H);
    for i=1:size(X,1)
        for k=1:R
            for h=1:H
                w_i=1;
                for j=1:ndims(X{i,k,1})
                    if j~=mod(iterNum,tensorDims)+1
                        w_i=w_i*norm(model(i,h).Proj{j,k}(:))^2; 
                    end
                end
                wCoeff(i,k,h)=1/(w_i+eps);
            end
        end
    end
    %calculating the X_current
    %samples set 1
%     X_current=cell(size(X,1),size(X,2));
    X_current2=cell(R,H);
    for l=1:R
        for h=1:H
            X_current2_temp=zeros(max(ntensorDim),c_num*length);
            for k=1:size(X,1)
    %            X_current_temp=zeros(ntensorDim(k),length);
               dims_temp=mod(iterNum,tensorDims)+1;
                   for j=1:length
                     %  if ~isempty(X{l,j})
                           samples_iter_sub=ttv(X{k,j},{model(k,h).Proj{1:dims_temp-1,l},model(k,h).Proj{dims_temp+1:ndims(X{k,1}),l}},[1:dims_temp-1,dims_temp+1:ndims(X{k,1})]); 
    %                        X_current_temp(:,j)=samples_iter_sub;
                           X_current2_temp(1:ntensorDim(k),(j-1)*c_num+k)=samples_iter_sub;
                     %  end
                   end
    %             X_current{k,l}=X_current_temp;
            end
        X_current2{l,h}=X_current2_temp;
        end
    end
    %samples set 2
    X_current2_2=cell(R,H);
    for l=1:R
        for h=1:H
            X_current2_temp=zeros(max(ntensorDim(:)),c_num*length);
            for k=1:size(X,1)
               dims_temp=mod(iterNum,tensorDims)+1;
                   for j=1:length
                      if Label(j)==k
                           samples_iter_sub=ttv(X{k,j},{model(k,h).Proj{1:dims_temp-1,l},model(k,h).Proj{dims_temp+1:ndims(X{k,l,1}),l}},[1:dims_temp-1,dims_temp+1:ndims(X{k,1})]); 
                           X_current2_temp(1:ntensorDim(k),(j-1)*c_num+1:j*c_num)=double(samples_iter_sub)*ones(1,c_num);
                      end
                   end
            end
            X_current2_2{l,h}=X_current2_temp;
        end
    end
    
    K_mat=zeros(alpha_num,alpha_num);
    for k=1:size(X,1)
        for l=1:R
            for h=1:H
               temp=zeros(1,length);
               temp(Label==k)=1;
               temp_vec=kron(temp,ones(1,c_num));
               temp_mat1=diag(temp_vec);
               temp=zeros(1,c_num);
               temp(k)=1;
               temp_vec=kron(ones(1,length),temp);
               temp_mat2=diag(temp_vec);
               temp=zeros(1,length);
               loc_temp=find(phi(k,:)==h);
               for i=1:size(loc_temp(:),1)
                temp(Label==loc_temp(i))=1;
               end
               temp_vec=kron(temp,ones(1,c_num));
               temp_mat3=diag(temp_vec);
               temp=zeros(1,c_num);
               temp(phi(k,:)==h)=1;
               temp_vec=kron(ones(1,length),temp);
               temp_mat4=diag(temp_vec);
               X_m_1=0.5*temp_mat4*temp_mat1*(X_current2_2{l,h}')*X_current2_2{l,h}*temp_mat1*temp_mat4;
               X_m_2=0.5*temp_mat3*temp_mat2*(X_current2{l,h}')*X_current2{l,h}*temp_mat2*temp_mat3;
               X_m_31=-0.5*temp_mat4*temp_mat1*(X_current2_2{l,h}')*X_current2{l,h}*temp_mat2*temp_mat3;
               X_m_32=-0.5*temp_mat3*temp_mat2*(X_current2{l,h}')*X_current2_2{l,h}*temp_mat1*temp_mat4;
               K_mat=K_mat+wCoeff(k,l,h)*(X_m_1+X_m_2+X_m_31+X_m_32);
            end
        end
    end
          tic
          object_F=@(alpha_v) alpha_v*K_mat*alpha_v'-2*sum(alpha_v);
          alpha_i_yi=zeros(c_num,length);
            for i=1:c_num
                temp=zeros(1,length);
                temp(Label==i)=1;
                alpha_i_yi(i,:)=temp;
            end
            alpha_i_yi=alpha_i_yi(:)';
          alpha_init = zeros(1,alpha_num);
            Aeq1=zeros(c_num,alpha_num);
            Aeq2=zeros(c_num,alpha_num);
            for i=1:c_num
                temp=zeros(1,length);
                temp(Label==i)=1;
                temp_vec=kron(temp,ones(1,c_num));
                Aeq1(i,:)=temp_vec;
                 temp=zeros(1,c_num);
                 temp(i)=1;
                 temp_vec=kron(ones(1,length),temp);
                 Aeq2(i,:)=temp_vec;
            end
            Aeq=[Aeq1-Aeq2];

            Aeq1=zeros(c_num*H,alpha_num);
            Aeq2=zeros(c_num*H,alpha_num);
            for i=1:c_num
                for h=1:H
                temp=zeros(1,length);
                temp(Label==i)=1;
                temp_h=ones(1,c_num);
                temp_h(phi(i,:)~=h)=0;
                temp_vec=kron(temp,temp_h);
                Aeq1((i-1)*H+h,:)=temp_vec;
                 temp=zeros(1,c_num);
                 temp(i)=1;
                 temp_h=zeros(1,length);
                loc_temp=find(phi(i,:)==h);
               for k=1:size(loc_temp(:),1)
                temp_h(Label==loc_temp(k))=1;
               end
                 temp_vec=kron(temp_h,temp);
                 Aeq2((i-1)*H+h,:)=temp_vec;
                end
            end
            Aeq=[Aeq1-Aeq2];

            C_f=C*ones(1,alpha_num);
            C_f(alpha_i_yi==1)=0;
          tic
          options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','iter','OptimalityTolerance',10^-12,'MaxIterations',1000);%'Display','iter', 'sqp','MaxIterations',1000000,'MaxFunctionEvaluations',1000000
          [alpha,fval,exitflag]=quadprog(2*K_mat,-2*ones(1,alpha_num),[],[],Aeq,zeros(c_num*H,1),zeros(1,alpha_num),C_f,[],options);
          toc
          fval
            alpha=alpha';
          object_F(alpha)
    for i=1:size(X,1)
        for j=1:R
            for h=1:H
                X_temp=X_current2{j,h};
                X_temp2=X_current2_2{j,h};
               temp=zeros(1,length);
               temp(Label==i)=1;
               temp_vec1=kron(temp,ones(1,c_num));
               temp=zeros(1,c_num);
               temp(i)=1;
               temp_vec2=kron(ones(1,length),temp);
               temp=zeros(1,length);
                loc_temp=find(phi(i,:)==h);
               for k=1:size(loc_temp(:),1)
                temp(Label==loc_temp(k))=1;
               end
               temp_vec3=kron(temp,ones(1,c_num));
               temp=zeros(1,c_num);
               temp(phi(i,:)==h)=1;
               temp_vec4=kron(ones(1,length),temp);
               temp=X_temp2*(temp_vec4.*temp_vec1.*alpha)'-X_temp*(temp_vec3.*temp_vec2.*alpha)';
                model(i,h).Proj{mod(iterNum,ndims(X{i,1}))+1,j}=wCoeff(i,j,h)*temp(1:ntensorDim(i));
            end
        end
    end
    if mod(iterNum,max(tensorDims))==0
         Iter_erroT=0;
         for i=1:size(X,1)
             for k=1:R
                 for h=1:H
                    Iter_erro=0;
                    for j=1:ndims(X{i,1})
                        Iter_erro=Iter_erro+norm(model(i,h).Proj{j,k}-model_temp(i,h).Proj{j,k})^2;
                    end
                    Iter_erroT=Iter_erroT+Iter_erro;
                 end
             end
        end
%         if iterNum==1
%         Iter_erroT=object_F(alpha);
%         end
%         Iter_erroCurrent=object_F(alpha);
%         Iter_erroT
        if Iter_erroT<tol2
            break;
        end
%         Iter_erroT=Iter_erroCurrent;
    end
        iterNum=iterNum+1;
        if iterNum>10
            break;
        end
end
%   for j=1:length
%      predict=X_current*(Par.w)'+Par.b 
%   end
distance_yi_m=zeros(c_num,c_num);
distance_yi_m_num=zeros(c_num,c_num);
distance_yi_m_BSV=zeros(c_num,c_num);
distance_yi_m_num_BSV=zeros(c_num,c_num);
distance_yi_m_NSV=zeros(c_num,c_num);
distance_yi_m_num_NSV=zeros(c_num,c_num);
for i=1:c_num
   temp_L=find(Label==i);
   for j=1:size(temp_L(:),1)
       temp=zeros(1,length);
       temp(temp_L(j))=1;
       temp_vec1=kron(temp,ones(1,c_num));
       SV=find(alpha>0.0001&alpha<C-0.0001&temp_vec1==1);
        NSV=find(alpha<=0.0001&temp_vec1==1&alpha_i_yi~=1);
        BSV=find(alpha>=C-0.0001&temp_vec1==1);
        if ~isempty(SV)
           for k=1:size(SV(:),1)
                [c_ind,l_ind]=ind2sub([c_num,length],SV(k));
                temp_yi=0;
                for l=1:R
                    temp_yi=temp_yi+ttv(X{i,l_ind},{model(i,phi(i,c_ind)).Proj{:,l}},[1:ndims(X{i,l_ind})]); 
                end
                temp_m=0;
                for l=1:R
                    temp_m=temp_m+ttv(X{c_ind,l_ind},{model(c_ind,phi(c_ind,i)).Proj{:,l}},[1:ndims(X{c_ind,l_ind})]); 
                end
                distance_yi_m(i,c_ind)=distance_yi_m(i,c_ind)+temp_yi-temp_m;
                distance_yi_m_num(i,c_ind)=distance_yi_m_num(i,c_ind)+1;
           end
        end
        if ~isempty(BSV)
            for k=1:size(BSV(:),1)
                [c_ind,l_ind]=ind2sub([c_num,length],BSV(k));
                 temp_yi=0;
                for l=1:R
                    temp_yi=temp_yi+ttv(X{i,l_ind},{model(i,phi(i,c_ind)).Proj{:,l}},[1:ndims(X{i,l_ind})]); 
                end
                temp_m=0;
                for l=1:R
                    temp_m=temp_m+ttv(X{c_ind,l_ind},{model(c_ind,phi(c_ind,i)).Proj{:,l}},[1:ndims(X{c_ind,l_ind})]); 
                end
                distance_yi_m_BSV(i,c_ind)=distance_yi_m_BSV(i,c_ind)+temp_yi-temp_m;
                distance_yi_m_num_BSV(i,c_ind)=distance_yi_m_num_BSV(i,c_ind)+1;
            end
        end
       if ~isempty(NSV)
            for k=1:size(NSV(:),1)
                [c_ind,l_ind]=ind2sub([c_num,length],NSV(k));
                 temp_yi=0;
                for l=1:R
                    temp_yi=temp_yi+ttv(X{i,l_ind},{model(i,phi(i,c_ind)).Proj{:,l}},[1:ndims(X{i,l_ind})]); 
                end
                temp_m=0;
                for l=1:R
                    temp_m=temp_m+ttv(X{c_ind,l_ind},{model(c_ind,phi(c_ind,i)).Proj{:,l}},[1:ndims(X{c_ind,l_ind})]); 
                end
                distance_yi_m_NSV(i,c_ind)=distance_yi_m_NSV(i,c_ind)+temp_yi-temp_m;
                distance_yi_m_num_NSV(i,c_ind)=distance_yi_m_num_NSV(i,c_ind)+1;
            end
        end
   end  
end
bp_Mat_temp=zeros(c_num*c_num,c_num*H);
for i=1:c_num
    for j=1:c_num
        if i==j
            continue;
        end
        bp_Mat_temp((i-1)*c_num +j,(i-1)*H+phi(i,j))=-1;
        bp_Mat_temp((i-1)*c_num +j,(j-1)*H+phi(j,i))=1;
%         bp_Mat_temp((i-1)*c_num +j,c_num*H+(i-1)*H+phi(i,j))=1;
    end
end
temp=eye(c_num);
bp_Mat=bp_Mat_temp(temp(:)==0,:);
% distance_bp=zeros((c_num-1)*c_num,1);
temp=eye(c_num);
distance_yi_m_num=distance_yi_m_num';
distance_yi_m_num_vec=distance_yi_m_num(temp==0);
distance_yi_m=distance_yi_m';
distance_yi_m_vec=distance_yi_m(temp==0);
distance_yi_m_num_BSV=distance_yi_m_num_BSV';
distance_yi_m_num_BSV_vec=distance_yi_m_num_BSV(temp==0);
distance_yi_m_BSV=distance_yi_m_BSV';
distance_yi_m_BSV_vec=distance_yi_m_BSV(temp==0);
distance_yi_m_num_NSV=distance_yi_m_num_NSV';
distance_yi_m_num_NSV_vec=distance_yi_m_num_NSV(temp==0);
distance_yi_m_NSV=distance_yi_m_NSV';
distance_yi_m_NSV_vec=distance_yi_m_NSV(temp==0);
% distance_bp(distance_yi_m_num_vec~=0)=distance_yi_m_vec(distance_yi_m_num_vec~=0)./distance_yi_m_num(distance_yi_m_num_vec~=0);
bp_Mat_SV=bp_Mat(distance_yi_m_num_vec~=0,:);
b_SV=distance_yi_m_vec(distance_yi_m_num_vec~=0)./distance_yi_m_num_vec(distance_yi_m_num_vec~=0)-2;
bp_Mat_BSV=bp_Mat(distance_yi_m_num_vec==0&distance_yi_m_num_BSV_vec~=0,:);
b_BSV=distance_yi_m_BSV_vec(distance_yi_m_num_vec==0&distance_yi_m_num_BSV_vec~=0)./distance_yi_m_num_BSV_vec(distance_yi_m_num_vec==0&distance_yi_m_num_BSV_vec~=0)-2;
bp_Mat_NSV=bp_Mat(distance_yi_m_num_vec==0&distance_yi_m_num_BSV_vec==0&distance_yi_m_NSV_vec~=0,:);
b_NSV=distance_yi_m_NSV_vec(distance_yi_m_num_vec==0&distance_yi_m_num_BSV_vec==0&distance_yi_m_NSV_vec~=0)./distance_yi_m_num_NSV_vec(distance_yi_m_num_vec==0&distance_yi_m_num_BSV_vec==0&distance_yi_m_NSV_vec~=0)-2;
if ~isempty(bp_Mat_SV)
    bp_Mat_SV=[bp_Mat_SV,eye(size(bp_Mat_SV,1)),-eye(size(bp_Mat_SV,1)),zeros(size(bp_Mat_SV,1),2*size(bp_Mat_NSV,1)+2*size(bp_Mat_BSV,1))];
    Aeq_SV=bp_Mat_SV;
    beq_SV=b_SV;
else
    Aeq_SV=[];
    beq_SV=[];
end
if ~isempty(bp_Mat_BSV)
    bp_Mat_BSV=-[bp_Mat_BSV,zeros(size(bp_Mat_BSV,1),2*size(bp_Mat_SV,1)),eye(size(bp_Mat_BSV,1)),-eye(size(bp_Mat_BSV,1)),zeros(size(bp_Mat_BSV,1),2*size(bp_Mat_NSV,1))];
    b_BSV=-b_BSV;
    A_BSV_NSV=bp_Mat_BSV;
    b_BSV_NSV=b_BSV;
else
    A_BSV_NSV=[];
    b_BSV_NSV=[];
end
if ~isempty(bp_Mat_NSV)
    bp_Mat_NSV=[bp_Mat_NSV,zeros(size(bp_Mat_NSV,1),2*(size(bp_Mat_BSV,1)+size(bp_Mat_SV,1))),eye(size(bp_Mat_NSV,1)),-eye(size(bp_Mat_NSV,1))];
    A_BSV_NSV=[A_BSV_NSV;bp_Mat_NSV];
    b_BSV_NSV=[b_BSV_NSV;b_NSV];
end
% options = optimoptions('linprog','Algorithm','interior-point');
f=[zeros(1,c_num*H),100*ones(1,2*(c_num-1)*c_num)];
lb=[zeros(1,2*c_num*H+(c_num-1)*c_num*2)];
size(f)
size(Aeq_SV)
size(A_BSV_NSV)
[bp,fval,existflag]=linprog(f,A_BSV_NSV,b_BSV_NSV,Aeq_SV,beq_SV,lb,[]);
b=zeros(H,c_num);
b(:)=bp(1:c_num*H);
b=b';
end
end
    