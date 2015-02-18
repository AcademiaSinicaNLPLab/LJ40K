function [y_processed] = mklv2_get_binary_vector(y, emotion)

for i=1:length(y)
    cmplen = min(length(y(i, :)), length(emotion));

    if strncmp(y(i, :), emotion, cmplen)
        y_processed(i) = 1;
    else
        y_processed(i) = -1;
    end
    
end

y_processed = transpose(y_processed);

