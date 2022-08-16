function L = Laplacian(features,num_neighbor)
% [~,C,~] = kmeans(features,size(features,1),'Distance','cosine','Replicate',100,'Display','final');
C = features;
A =zeros(size(C,1),size(C,1));
Degree= zeros(size(A));
Mdl = KDTreeSearcher(C);
for i =1:size(C,1) 
    [n,d] = knnsearch(Mdl,features(i,:),'k',num_neighbor);
    for j=2:size(n,2)
%         A(i,n(1,j))=1/(1+exp(-d(1,j)));
        A(i,n(1,j))=1;
    end
end
for i= 1:size(Degree,1)
    Degree(i,i)=sum(A(i,:),2);
end
% Identity = diag(ones(1,size(features,1)));
% L = Identity - Degree^(-1/2)*A*Degree^(-1/2);
L = Degree^(-1/2)*A*Degree^(-1/2);
end

