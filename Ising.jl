using GraphicalModelLearning
using LinearAlgebra
using StatsBase
using JSON
using FFTW
using Distributions
using CSV
using DataFrames
using Plots


function DCT(image,quality)
    # image: in binary form [-1,1]
    # quality: required percentage of the original image
    # reconstructed_image: in binary form [-1,1]

    g = image*128
    
    # Compression (dct)
    G = dct(g,1)
    R = dct(G,2)

    # Expansion:
    X = R[:]
    X = X[:,:]'

    # calculate cut-off coefficient
    ind = sortperm(abs.(vec(X)), rev=true)
    coeffs = 1
    while (norm(X[ind[1:coeffs]])/norm(X))^2 < quality
        coeffs += 1
    end
    
    # reconstruct from coeffs > cut-off coeff
    R[abs.(R) .< abs.(X[ind[coeffs]])] .= 0
    
    # inverse dct
    S = idct(R,2)
    T = idct(S,1)

    reconstructed_image = sign.(T .+ 0.000001)
    
    return reconstructed_image

end 

function To_histogram(data_set)  

    ans_dict = countmap(collect(eachcol(data_set')))
    vectors = collect(keys(ans_dict))
    freq = get.([ans_dict],vectors,"na")
    histogram = Int.([freq reduce(hcat,vectors)'])
    return histogram
end

function MCMC_func(A, S, T)
    # A: network matrix    
    # T: num of time steps
    # S: initial samples (each column is a sample)

    n = size(S,1);
    m = size(S,2);

    D = Diagonal(diag(A))
    B = A - D

    # MCMC samples generation
    for s=1:m
        for t=1:T-1
            i = rand(1:n) 
            v = 2*sum(B[i,:].*S[:,s]) + 2*D[i,i]
            p = exp(v)/(1 + exp(v))   # prob. of spin=1
            spin = rand() < p
            spin = spin*2 - 1
            S[i,s] = spin 
        end 
    end
    return S
end

function moments(S)

    N = sum(S[:,1])
    m1 = sum(S[:,1].*S[:,2:end], dims=1)/N  
    
    
    m2 = zeros(size(S,2)-1,size(S,2)-1);
    for i = 1:size(S,1)
        v = S[i,2:end][:,:]-m1'
        m2 = m2 + S[i,1].*(v*v')
    end
    m2 = m2/(N-1)

    return m1, m2
    
end

function moment_error(S1, S2)

    (m1, m2) = moments(S1)
    (m1_, m2_) = moments(S2)
    
    error_m1 = maximum(abs.(m1-m1_))
    error_m2 = maximum(abs.(m2-m2_))

    return error_m1, error_m2
end  

function Total_Variation(S,Sj)
    # S: all configurations with their prob (original distribution). 
    # Sj: samples distribution to be measured.

    P_Sj = Sj[:,1]/sum(Sj[:,1]);

    # for each row in Sj, find corresp. location in S, then calculate difference in prob.
    idx = zeros(size(Sj,1))
    for i = 1:size(Sj,1)
        idx[i] = findfirst(vec(all(S[:,2:end] .== Sj[i,2:end][:,:]', dims=2)))
    end
    TV1 = sum(abs.(S[Int.(idx),1] - P_Sj))/2                  

    # calculate prob. difference for rows that are not in Sj
    idx_ = setdiff(1:size(S,1),idx)                    
    TV2 = sum(abs.(S[Int.(idx_),1]))/2

    TV = TV1 + TV2

    return TV
end

function repelem(A, v)
    v = Int.(v)
    row_idx = inverse_rle(axes(v, 1), v)
    return A[row_idx, :]
end


AA = [0     1     0     0    -1     0     0     0     0     0     0     0     0     0     0     0;
      1     0     1     0     0     1     0     0     0     0     0     0     0     0     0     0;
      0     1     0     1     0     0     1     0     0     0     0     0     0     0     0     0;
      0     0     1     0     0     0     0     1     0     0     0     0     0     0     0     0;
     -1     0     0     0     0     1     0     0    -1     0     0     0     0     0     0     0;
      0     1     0     0     1     0     1     0     0     1     0     0     0     0     0     0;
      0     0     1     0     0     1     0    -1     0     0     1     0     0     0     0     0;
      0     0     0     1     0     0    -1     0     0     0     0     1     0     0     0     0;
      0     0     0     0    -1     0     0     0     0     1     0     0     1     0     0     0;
      0     0     0     0     0     1     0     0     1     0     1     0     0     1     0     0;
      0     0     0     0     0     0     1     0     0     1     0     1     0     0     1     0;
      0     0     0     0     0     0     0     1     0     0     1     0     0     0     0    -1;
      0     0     0     0     0     0     0     0     1     0     0     0     0     1     0     0;
      0     0     0     0     0     0     0     0     0     1     0     0     1     0    -1     0;
      0     0     0     0     0     0     0     0     0     0     1     0     0    -1     0     1;
      0     0     0     0     0     0     0     0     0     0     0    -1     0     0     1     0]


################## Brute-Force Sampling (generate S1) ##################
n = 16          # n of nodes
N = 1000000      # n of samples 
sigma = [-1,1]

# all possible combination vectors (each column of m)
autof = fill(sigma, n)
a = vec(collect(Base.product(autof...)))
m = reshape(reinterpret(Int, a), (n,:))   

# probability of each combination vector
H = (-m[1,:].*m[5,:] - m[5,:].*m[9,:] + m[9,:].*m[13,:] 
    + m[2,:].*m[6,:] + m[6,:].*m[10,:] + m[10,:].*m[14,:] 
    + m[3,:].*m[7,:] + m[7,:].*m[11,:] + m[11,:].*m[15,:] 
    + m[4,:].*m[8,:] + m[8,:].*m[12,:] - m[12,:].*m[16,:] 
    + m[1,:].*m[2,:] + m[5,:].*m[6,:] + m[9,:].*m[10,:] + m[13,:].*m[14,:]
    + m[2,:].*m[3,:] + m[6,:].*m[7,:] + m[10,:].*m[11,:] - m[14,:].*m[15,:]
    + m[3,:].*m[4,:] - m[7,:].*m[8,:] + m[11,:].*m[12,:] + m[15,:].*m[16,:])
P = exp.(H)./sum(exp.(H)) 
# S = [P[:,:] m']

# sample generation with exact prob.
s = rand(Multinomial(N, P))
S1 = [s m']


A = learn(S1)
println(maximum(abs.(A-AA)))

################################### Moments errors (w & w/o MCMC correction) for varied levels of compression  ###############################################

preserved_quality = 0.1:0.2:0.9
m_error_compress = zeros(length(preserved_quality),2)
m_error_MCMC = zeros(length(preserved_quality),2)


for k=1:length(preserved_quality)

# DCT compress & expand 
S2 = copy(S1)
for i=1:size(S1,1)
    image = S1[i,2:end][:,:]'
    reconstructed_image = DCT(image,preserved_quality[k])
    S2[i,2:end] = reconstructed_image
end
S2_full = repelem(S2[:,2:end],S2[:,1])
S2 = To_histogram(S2_full)
m_error_compress[k,:] = collect(moment_error(S1,S2))



# run MCMC with learned model
S3_full = MCMC_func(A, S2_full', 10*16)'
S3 = To_histogram(S3_full)
m_error_MCMC[k,:] = collect(moment_error(S1,S3))

end


plot(1 .- preserved_quality, m_error_compress[:,2], markershape = :circle, label="DCT Compression")
plot!(1 .- preserved_quality,m_error_MCMC[:,2],  markershape = :circle, label="Super Resolution", 
      xticks = ([0.1,0.3,0.5,0.7,0.9]), titlefont=font(13), xtickfontsize=11, ytickfontsize=11)
xlabel!("Compression (C)")
ylabel!("Max element-wise moment error (E)")

#########################################################################################################################


# run MCMC with learned model from scratch
S_full = MCMC_func(A, rand(n,N), 250*16)'
collect(moment_error(S1,To_histogram(S_full)))

