function [kernelcellaux,kerneloptioncellaux,variablecellaux]=CreateKernelListWithVariable(variablecell,dim,kernelcell,kerneloptioncell,feature_start_idx)


j=1;
for i=1:length(variablecell)
    optnum = str2num(variablecell{i});
    if size(optnum) == [1 1]
        varopt = 'number';
    else
        varopt = variablecell{i};
    end

    switch varopt
        case 'all'
            kernelcellaux{j}=kernelcell{i};
            kerneloptioncellaux{j}=kerneloptioncell{i};
            variablecellaux{j}=1:dim;
            j=j+1;    
        case 'single'
            for k=1:dim
                kernelcellaux{j}=kernelcell{i};
                kerneloptioncellaux{j}=kerneloptioncell{i};
                variablecellaux{j}=k;
                j=j+1;
            end;    
    	case 'random'
    		kernelcellaux{j}=kernelcell{i};
            kerneloptioncellaux{j}=kerneloptioncell{i};
    		indicerand=randperm(dim);
    		nbvarrand=floor(rand*dim)+1;         
       		variablecellaux{j}=indicerand(1:nbvarrand);
            j=j+1;
        case 'number'
            kernelcellaux{j}=kernelcell{i};
            kerneloptioncellaux{j}=kerneloptioncell{i};
            n_feature_type = size(feature_start_idx, 2);
            if n_feature_type > optnum              % middle
                variablecellaux{j}=feature_start_idx(optnum):feature_start_idx(optnum+1)-1;
            elseif n_feature_type == optnum         % last
                variablecellaux{j}=feature_start_idx(optnum):dim;
            else
                error('incorrect feature idx');
            end             
            j=j+1; 
    end;
end;
variablecellaux