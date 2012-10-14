% ---- code to generate data replicating Zorzi's paper ----
D = zeros(30, 30, 8*32*200);
randn('seed', 100);
index = 1;
for N = 1:32
    for ia = 1:8
        % -- choose cumulative surface area --
        
        cumA = 32 + 32*(randi([1, 8])-1); 
        it = 1
        while it < 201
            count = 0; 
            tries = 0
            run_cumA = cumA;
            while run_cumA > 0 && N > count
                % -- caculate sides of the rectangle --
                
                A = run_cumA/(N-count) + 0.15*randn;
                
                a = sqrt(A); 
                
                h = round(a + 0.3*randn);
                w = round(a + 0.3*randn);
                if h ==0
                    h = 1;
                end
                if w ==0
                    w = 1;
                end
                
                posh = randi([1, 30-h+1]);
                posw = randi([1, 30-w+1]); 
                tries = tries + 1; 
                if tries > 300
                    break
                    it = it -1;
                end
                if ~check_overlap(D(:, :, index), h, w, posh, posw)
                    D(posh:posh+h-1, posw:posw+w-1, index) = 1;
                    run_cumA = run_cumA - h*w;
                    count = count + 1;
                    fprintf('h : %d, w : %d\n', h, w);                    
                end                
            end
            cumA
            count
            fprintf('processed %dth image\n', index); 
            %       imagesc(D(:, :, index)); pause(0.01);            
            index = index + 1;
            it = it +1;
        end
    end
end
