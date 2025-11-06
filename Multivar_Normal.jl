using LinearAlgebra
using Distributions
using JuMP
using Ipopt
using FFTW
using Plots

function  Multivariate_Normal_func(N,n_samples)
    
    
    Mu = (rand(N,) .- 0.5)
    Sigma = [1 0;0 0]
    while cond(Sigma)>50
        Rnd = (rand(N,N) .- 0.5)
        Sigma = Rnd*Rnd'
    end
    d = MvNormal(Mu, Sigma)
    
    # n_samples = 20000
    x = rand(d, n_samples)'
    
    A_model = -0.5*inv(Sigma)
    b_model = inv(Sigma)'Mu

    # println(cond(Sigma))
    # println(eigvals(Sigma))

    return x, A_model, b_model
end

function learn_score_matching(x)
    model = Model(Ipopt.Optimizer)
    @variable(model, A[1:N,1:N])
    @variable(model, b[1:N])
    @objective(model, Min, 
                sum(sum(2*A[i,i] + 0.5*(2*A[i,:]'x[:,k] + b[i])^2  for i=1:N) for k=1:n_samples)/n_samples 
            )
    optimize!(model)
    A_est = JuMP.value.(A)
    b_est = JuMP.value.(b)
    
    return A_est, b_est
end

function model_error(A_model, b_model, A_est, b_est)
    A_err = maximum(abs.(A_est - A_model))/maximum(abs.(A_model))
    b_err = maximum(abs.(b_est - b_model))/maximum(abs.(b_model))
    return A_err, b_err
end

function Langevin_Dynamics(x0,A,b,T)
    dt = 0.05
    N = size(A,1)
    x_traj = zeros(N,T)
    x = x0
    for k=1:T
        x = x + dt*(2*A'x + b) + sqrt(2*dt)*randn(N)
        x_traj[:,k] = x
    end
    return x_traj
end

function moments(S)

    N = size(S,1)
    m1 = sum(S, dims=1)/N  
    
    
    m2 = zeros(size(S,2),size(S,2));
    for i = 1:N
        v = S[i,:][:,:]-m1'
        m2 = m2 + (v*v')
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



# Sampling from Multivariate Normal 
N = 16
n_samples = 100000
S1, A, b = Multivariate_Normal_func(N,n_samples)


# Learning distribution (using score-matching)
A_est, b_est = learn_score_matching(S1')


################################################################################################################

preserved_quality = 0.1:0.2:0.9
m_error_compress = zeros(length(preserved_quality),2)
m_error_MCMC = zeros(length(preserved_quality),2)


for k=1:length(preserved_quality)

# compress & expand 
S2 = copy(S1)
for i=1:n_samples
    image = S1[i,:][:,:]'
    reconstructed_image = DCT(image,preserved_quality[k])
    S2[i,:] = reconstructed_image
end
m_error_compress[k,:] = collect(moment_error(S1, S2))

  
# run MCMC with learned model
S3 = copy(S2)
for i=1:n_samples
    S3[i,:] = Langevin_Dynamics(S2[i,:], A_est, b_est, 7*N)[:,end]
end
m_error_MCMC[k,:] = collect(moment_error(S1, S3))

end

# #########################################################################################################################
plot(1 .- preserved_quality, m_error_compress[:,2], markershape = :circle, label="DCT Compression")
plot!(1 .- preserved_quality,m_error_MCMC[:,2],  markershape = :circle, label="Super Resolution", 
      xticks = ([0.1,0.3,0.5,0.7,0.9]), titlefont=font(13), xtickfontsize=11, ytickfontsize=11)
xlabel!("Compression (C)")
ylabel!("Max element-wise moment error (E)")

#########################################################################################################################


