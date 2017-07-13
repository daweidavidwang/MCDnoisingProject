epsilon = 0.05;
maxrange = 3000;
ext = { '*.exr'};
folder = 'E:\datasetcombine\GT\';
distribute = zeros(1,maxrange/epsilon);
filepath = [];
for i = 1 : length(ext)
    filepath = cat(1,filepath, dir(fullfile(folder, ext{i})));
end

for i = 1 : length(filepath)
    i = i
    image = exrread(fullfile(folder,filepath(i).name));
    image = reshape(image,[1,size(image,1)*size(image,2)*size(image,3)]);
    for j = 1 : size(image,2)
        distribute(1,floor(image(1,j)/epsilon)+1) = distribute(1,floor(image(1,j)/epsilon)+1)+1;
    end
end

x = 0:(maxrange/epsilon)-1;
x = x.*epsilon;


plot(x,distribute);
integral = sum(distribute(:));
sump = integral;
i = length(x);
while i > 0 
    sump = sump - distribute(1,i);
    if(sump < 0.99*integral)
        result = i * epsilon
        break;
    end
    i = i-1;
end
i = length(x);
 while i > 0 
     if(distribute(1,i)>0)
         resultppp = i *epsilon
         break;
     end
     i = i-1;
 end

    