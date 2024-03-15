@testset "Quac" begin
    using Quac

    @testset "QFT" begin
        n = 3
        qftcirc = Quac.Algorithms.QFT(n)
        qftqtn = Quantum(qftcirc)

        # correct number of inputs and outputs
        @test ninputs(qftqtn) == n
        @test noutputs(qftqtn) == n
        @test socket(qftqtn) == Operator()

        # all open indices are sites
        siteinds = getindex.((qftqtn,), sites(qftqtn))
        @test issetequal(inds(TensorNetwork(qftqtn), :open), siteinds)

        # all inner indices are not sites
        # TODO too strict condition. remove?
        notsiteinds = setdiff(inds(TensorNetwork(qftqtn)), siteinds)
        @test_skip issetequal(inds(TensorNetwork(qftqtn), :inner), notsiteinds)
    end

    @testset "evolve" begin
        n = 10
        timesteps = 20
        δₜ = 0.1
        ket₊ = 1/√2 * [1, 1]
        observables = Dense.([Z(5)])

        function trotter_XX(i,j; δₜ=δₜ)
            mat = kron(Matrix(X()), Matrix(X()))
            mat = cis(δₜ * mat)
            mat = reshape(mat, 2,2,2,2)
            Dense(Qrochet.Operator(), mat; sites=[Site(i), Site(j), Site(i, dual=true), Site(j, dual=true)])
        end

        function trotter_Z(i; λ, δₜ=δₜ)
            mat = Matrix(Z())
            mat = cis(- λ * δₜ * mat)

            Dense(Qrochet.Operator(), mat; sites=[Site(i), Site(i, dual=true)])
        end

        @testset "not canonical" begin
            ψ = convert(Chain, Product(fill(ket₊, n)))

            for it in 1:timesteps
                for (i,j) in Iterators.filter(==(2) ∘ length, Iterators.partition(1:n,2))
                    evolve!(ψ, trotter_XX(i,j; δₜ))
                end
                for (i,j) in Iterators.filter(==(2) ∘ length, Iterators.partition(2:n,2))
                    evolve!(ψ, trotter_XX(i,j; δₜ))
                end
                for i in 1:n
                    evolve!(ψ, trotter_Z(i; λ=0.3, δₜ))
                end
            end
        end

        @testset "canonical" begin
            ψ = convert(Chain, Product(fill(ket₊, n)))
            canonize!(ψ)

            for it in 1:timesteps
                for (i,j) in Iterators.filter(==(2) ∘ length, Iterators.partition(1:n,2))
                    evolve!(ψ, trotter_XX(i,j; δₜ); iscanonical=true)
                end
                for (i,j) in Iterators.filter(==(2) ∘ length, Iterators.partition(2:n,2))
                    evolve!(ψ, trotter_XX(i,j; δₜ); iscanonical=true)
                end
                for i in 1:n
                    evolve!(ψ, trotter_Z(i; λ=0.3, δₜ);  iscanonical=true)
                end
            end

            # Without truncation, we expect the norm and canonicity to be preserved
            @test norm(ψ) ≈ 1.0
            @test isleftcanonical(ψ, Site(1))
            for i in 2:n
                contracted = contract(ψ, :between, Site(i - 1), Site(i); direction = :right)
                @test isleftcanonical(contracted, Site(i); atol=1e-10)
            end
        end
    end
end
