function [g,Q] = hist_match(Z,h,bins);

% input:    Z - data that needs to be adjusted
%           h - target histogram
%           t - h's bins 
% output:   g - matching function Q=g(Z)
%           Q - adjusted Z with cumulative histogram corresponding to
%           cumsum(h)

L=length(bins);
g=NaN*ones(1,L);
cdf=cumsum(h);

hQ=hist(Z,bins);
cdfQ=cumsum(hQ);

c=0; kQ=1;
while c<sum(hQ) & kQ<L-1
    kQ=min(find(cdfQ>c));
    k=max(find(cdf<cdfQ(kQ)));
    g(kQ)=bins(k+1);
    kQ=kQ+1;
    c=cdfQ(kQ);
end

ind=find(isfinite(g)==1);
ind_empty=find(isfinite(g)~=1);
g(ind_empty) = interp1(bins(ind),g(ind),bins(ind_empty));

p=polyfit(bins,1:L,1);
idx=round(polyval(p,Z));
Q=g(idx);

