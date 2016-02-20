function adagrad(train)
%  Solves L2 regularized logisitc regression using Adagrad
%  f(w) = 1/m {\sum_i log(1+exp(-y_iw'*x_i)) + \rho/2*\|w\|_2^2} (avg loss +
%  regularization term)
%  We solve the above problem using Adagrad algorithm by Duchi et al.
%  Input: file name , only libsvm format supported till now.
%  Output: convergence plot of SGD on L2 regularised logisitc regression
%  Contact:
% Chandresh Maurya
% Email: ckm.jnu@gmail.com

clc
filepath=sprintf('/home/chandu-pc/Dropbox/data/%s',train);
[y,x]=libsvmread(filepath);
% feature scaling
% v=max(abs(x));
% x=bsxfun(@rdivide,x,v);
% clear v;

m = length(y); % store the number of training examples
x = [ ones(m,1) x]; % Add a column of ones to x
n = size(x,2); % number of features
print_step = 20;
printout = floor(m/print_step);
% set global learning rate
eta = 0.001;

rho_v = [1];

for l=1:length(rho_v)
    rho = rho_v(l);
   
 for kkk=1:1
     
    samples_seen = zeros(n,1);
    k = 1;
    w   = zeros(n,1);
    obj = zeros(print_step,print_step);
    ind = zeros(print_step,1);
    G   = zeros(n,1);% running sum of squares of the component-wise gradients
    
	%stochastic gradient descent - loop over one training set at a time
	  for t = 1:m		
%         sample index j uniformaly at random
        j = randi(m);
        samples_seen(t) = j;
%         evaluate the gradient at x(j,:)
        grad = eval_grad(w,y(j),x(j,:),rho);
%        calculate the diag aprox of sum of squares of gradients
        G = G + grad.^2;
        
%       update paramter vector
        w = w - eta*(((G + 1e-8).^-0.5).*grad);
%         w(2:end) = (1- eta*rho)*w(2:end); %% don't penalize the bias term
%         From time to time, evaluate the objective function to see the
%         progress
        if(mod(t,printout) == 0)
         obj(kkk,k)=  eval_obj(w,y(samples_seen(1:t)),x(samples_seen(1:t),:),rho) ;
         ind(k)= t;
         k = k+1;
        end
        
    end
  
end
plot(ind,mean(obj));
pause;
end

end


function  grad = eval_grad(w,y,x,rho)
         prod = -y*(x*w); % y is scalar here
         grad = -((exp(prod) / (1+exp(prod)))*y)*(x') + rho*w ;
   
end

function obj=eval_obj(w,y,x,rho) 
prod = -y.*(x*w); % y is vector here
obj = sum(log(1+exp(prod))) + 0.5*rho*norm(w)^2;
obj = obj / length(y);% online avg of objective function 
end
