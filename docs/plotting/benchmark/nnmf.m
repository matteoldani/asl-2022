input_sizes = [100, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000];
k = 3;

for k1 = input_sizes
    A = rand(k1,k1);
    fprintf('size(A) is %s\n', mat2str(size(A)));
    tic
        [W,H] = nnmf(A,k, 'Options',statset('MaxIter',1000,'TolFun',0));
    toc
end



