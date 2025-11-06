using LinearAlgebra
using Distributions 
using Random
using StatsBase
using Plots
using JuMP
using Ipopt
using FFTW
using JLD2

using Graphs
using GraphPlot 

function meshgrid(x,y)
    X = x' .* ones(length(y))
    Y = ones(length(x))' .* y
    return X,Y
end 

function expand(S)
    mat = S[:,2:end] # matrix 
    v = Int.(S[:,1])       # freq of each rows 
    
    row_idx = inverse_rle(axes(v, 1), v)
    S_full = mat[row_idx, :]

    return S_full
end

function score_func(x,A,p)
    grad_log_density = -4*p[1]*x.^3 - 2*(p[2]*I + p[3]*A)*x
    return grad_log_density
end

function derivative_2nd(x,A,p)
    grad_square = -12*p[1]*x.^2 - 2*diag(p[2]*I + p[3]*A)
    return grad_square
end

function Langevin_Dynamics(x0,A,p,T)
    dt = 0.01
    N = size(A,1)
    x_traj = zeros(N,T)
    x = x0
    for k=1:T
        x = x + dt*score_func(x,A,p) + sqrt(2*dt)*randn(N)
        x_traj[:,k] = x
    end
    return x_traj
end

function Langevin_Dynamics_2(x0,A,p,T)
    dt = 0.001
    N = size(A,1)

    T_burn = 50000
    x_burn = zeros(N,T_burn)
    x = x0
    for k=1:T_burn
        x = x + dt*score_func(x,A,p) + sqrt(2*dt)*randn(N)
        x_burn[:,k] = x
    end

    x_traj = zeros(N,T)
    x = x_burn[:,end]
    for k=1:T
        x = x + dt*score_func(x,A,p) + sqrt(2*dt)*randn(N)
        x_traj[:,k] = x
    end

    return x_burn, x_traj
end

function learn_score_matching(x)
    model = Model(Ipopt.Optimizer)

    @variable(model, A[1:N,1:N])    # contains p[2:3] params
    @variable(model, a)             # p[1] params


    @objective(model, Min, 
                sum(  sum(-12*a*x[:,k].^2 - 2*diag(A)) + sum(0.5*(-4*a*x[:,k].^3 - 2*A*x[:,k]).^2)  for k=1:n_samples)/n_samples 
            )


    optimize!(model)
    A_est = JuMP.value.(A)
    a_est = JuMP.value.(a)
    return A_est, a_est
end

function learn_score_matching_2(x, AA)
    N = size(x,1)
    n_samples = size(x,2)
    model = Model(Ipopt.Optimizer)

    @variable(model, a)             # p[1] params
    @variable(model, b)             # p[2] params
    @variable(model, c)             # p[3] params

    A = b*I(N) .+ c*AA
    @objective(model, Min, 
                sum(  sum(-12*a*x[:,k].^2 - 2*diag(A)) + sum(0.5*(-4*a*x[:,k].^3 - 2*A*x[:,k]).^2)  for k=1:n_samples)/n_samples 
            )


    optimize!(model)
    a_est = JuMP.value.(a)
    b_est = JuMP.value.(b)
    c_est = JuMP.value.(c)
    return a_est, b_est, c_est
end

function learn_errors(A_est,b_est,d_est,A,b,d)
    e1 = maximum(abs.(A_est-A))
    e2 = maximum(abs.(b_est-b))
    e3 = maximum(abs.(d_est-d))

    errors = [e1,e2,e3]
    return errors
end

function To_histogram(data_set)  

    ans_dict = countmap(collect(eachcol(data_set')))
    vectors = collect(keys(ans_dict))
    freq = get.([ans_dict],vectors,"na")
    histogram = [freq reduce(hcat,vectors)']
    return histogram
end

function moments(S)

    m1 = sum(S, dims=1)/size(S,1)
    
    m2 = zeros(size(S,2),size(S,2));
    for i = 1:size(S,1)
        v = S[i,:][:,:]-m1'
        m2 = m2 + v*v'
    end
    m2 = m2/(size(S,1)-1)

    return m1, m2
    
end

function moment_error(S1, S2)

    (m1, m2) = moments(S1)
    (m1_, m2_) = moments(S2)
    
    error_m1 = maximum(abs.(m1-m1_))
    error_m2 = maximum(abs.(m2-m2_))

    return error_m1, error_m2
end  

function DCT(image,quality)
    # image: in binary form [-1,1]
    # quality: required percentage of the original image
    # reconstructed_image: in binary form [-1,1]

    im_max = maximum(abs.(image))
    g = image*128/im_max
    
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

    reconstructed_image = T*im_max/128
    
    return reconstructed_image

end 

function QoIs(S,A)
    q1 = 0
    q2 = 0
    q3 = 0
    n_samples = size(S,1)
    for i = 1:n_samples
        s = S[i,:]
        q3 += (1/24)*(s'A's)/2
        q2 += sum(s.^2)/16
        q1 += sum(s.^4)/16
    end
    return q1/n_samples, q2/n_samples, q3/n_samples 
end

################################# Sample Generation ################################################
# create graph/adj_matrix on lattice  
g = Graphs.SimpleGraphs.grid((4,4))
gplot(g, nodelabel=1:nv(g))
A = adjacency_matrix(g)

error_compress = zeros(5,)
error_MCMC = zeros(5,)

p = [0.1, 0.2, 0.3] + 0.02*rand(3,) # network parameters

# Sampling with Langevin_Dynamics
N = 16
n_samples = 100000
(x_burn, x_traj) = Langevin_Dynamics_2(zeros(N,),A,p,100*n_samples)
S1 = x_traj[:,1:100:end]'

################################################# Learn model ######################################################################

# model learning 
p_est = collect(learn_score_matching_2(S1',A))
qois_S1 = collect(QoIs(S1,A))

################################################ Compression & MCMC correction #########################################################################
preserved_quality = 0.1:0.2:0.9
for k=1:length(preserved_quality)

    # compress & expand 
    S2 = copy(S1)
    for i=1:size(S1,1)
        image = S1[i,:][:,:]'
        reconstructed_image = DCT(image,preserved_quality[k])
        S2[i,:] = reconstructed_image
    end
    qois_S2 = collect(QoIs(S2,A))
    error_compress[k] = maximum(abs.(qois_S1 - qois_S2))

    # run MCMC with learned model
    S3 = copy(S2)
    for i=1:size(S2,1)
        S3[i,:] = Langevin_Dynamics(S2[i,:],A,p_est,16*20)[:,end]
    end
    qois_S3 = collect(QoIs(S3,A))
    error_MCMC[k] = maximum(abs.(qois_S1 - qois_S3))

end



#################################################################################
plot(1 .- preserved_quality, m_error_compress[:,2], markershape = :circle, label="DCT Compression")
plot!(1 .- preserved_quality,m_error_MCMC[:,2],  markershape = :circle, label="Super Resolution", 
      xticks = ([0.1,0.3,0.5,0.7,0.9]), titlefont=font(13), xtickfontsize=11, ytickfontsize=11)
xlabel!("Compression (C)")
ylabel!("Max element-wise QoIs error (E)")
