@testset "Chain ansatz" begin
    qtn = Chain(State(), Periodic(), [rand(2, 4, 4) for _ in 1:3])
    @test socket(qtn) == State()
    @test ninputs(qtn) == 0
    @test noutputs(qtn) == 3
    @test issetequal(sites(qtn), [site"1", site"2", site"3"])
    @test boundary(qtn) == Periodic()
    @test leftindex(qtn, site"1") == rightindex(qtn, site"3") != nothing

    qtn = Chain(State(), Open(), [rand(2, 2), rand(2, 2, 2), rand(2, 2)])
    @test socket(qtn) == State()
    @test ninputs(qtn) == 0
    @test noutputs(qtn) == 3
    @test issetequal(sites(qtn), [site"1", site"2", site"3"])
    @test boundary(qtn) == Open()
    @test leftindex(qtn, site"1") == rightindex(qtn, site"3") == nothing

    qtn = Chain(Operator(), Periodic(), [rand(2, 2, 4, 4) for _ in 1:3])
    @test socket(qtn) == Operator()
    @test ninputs(qtn) == 3
    @test noutputs(qtn) == 3
    @test issetequal(sites(qtn), [site"1", site"2", site"3", site"1'", site"2'", site"3'"])
    @test boundary(qtn) == Periodic()
    @test leftindex(qtn, site"1") == rightindex(qtn, site"3") != nothing

    qtn = Chain(Operator(), Open(), [rand(2, 2, 4), rand(2, 2, 4, 4), rand(2, 2, 4)])
    @test socket(qtn) == Operator()
    @test ninputs(qtn) == 3
    @test noutputs(qtn) == 3
    @test issetequal(sites(qtn), [site"1", site"2", site"3", site"1'", site"2'", site"3'"])
    @test boundary(qtn) == Open()
    @test leftindex(qtn, site"1") == rightindex(qtn, site"3") == nothing

    @testset "Site" begin
        using Qrochet: leftsite, rightsite
        qtn = Chain(State(), Periodic(), [rand(2, 4, 4) for _ in 1:3])

        @test leftsite(qtn, Site(1)) == Site(3)
        @test leftsite(qtn, Site(2)) == Site(1)
        @test leftsite(qtn, Site(3)) == Site(2)

        @test rightsite(qtn, Site(1)) == Site(2)
        @test rightsite(qtn, Site(2)) == Site(3)
        @test rightsite(qtn, Site(3)) == Site(1)

        qtn = Chain(State(), Open(), [rand(2, 2), rand(2, 2, 2), rand(2, 2)])

        @test leftsite(qtn, Site(1)) |> isnothing
        @test rightsite(qtn, Site(3)) |> isnothing

        @test leftsite(qtn, Site(2)) == Site(1)
        @test leftsite(qtn, Site(3)) == Site(2)

        @test rightsite(qtn, Site(2)) == Site(3)
        @test rightsite(qtn, Site(1)) == Site(2)
    end

    @testset "truncate" begin
        qtn = Chain(State(), Open(), [rand(2, 2), rand(2, 2, 2), rand(2, 2)])
        canonize_site!(qtn, Site(2); direction = :right, method = :svd)

        @test_throws Qrochet.MissingSchmidtCoefficientsException truncate!(qtn, [Site(1), Site(2)]; maxdim = 1)
        @test_throws ArgumentError truncate!(qtn, [Site(2), Site(3)])

        truncated = Qrochet.truncate(qtn, [Site(2), Site(3)]; maxdim = 1)
        @test size(TensorNetwork(truncated), rightindex(truncated, Site(2))) == 1
        @test size(TensorNetwork(truncated), leftindex(truncated, Site(3))) == 1

        singular_values = select(qtn, :between, Site(2), Site(3))
        truncated = Qrochet.truncate(qtn, [Site(2), Site(3)]; threshold = singular_values[2] + 0.1)
        @test size(TensorNetwork(truncated), rightindex(truncated, Site(2))) == 1
        @test size(TensorNetwork(truncated), leftindex(truncated, Site(3))) == 1
    end

    @testset "rand" begin
        using LinearAlgebra: norm

        @testset "State" begin
            n = 8
            χ = 10

            qtn = rand(Chain, Open, State; n, p = 2, χ)
            @test socket(qtn) == State()
            @test ninputs(qtn) == 0
            @test noutputs(qtn) == n
            @test issetequal(sites(qtn), map(Site, 1:n))
            @test boundary(qtn) == Open()
            @test isapprox(norm(qtn), 1.0)
            @test maximum(last, size(TensorNetwork(qtn))) <= χ
        end

        @testset "Operator" begin
            n = 8
            χ = 10

            qtn = rand(Chain, Open, Operator; n, p = 2, χ)
            @test socket(qtn) == Operator()
            @test ninputs(qtn) == n
            @test noutputs(qtn) == n
            @test issetequal(sites(qtn), vcat(map(Site, 1:n), map(adjoint ∘ Site, 1:n)))
            @test boundary(qtn) == Open()
            @test isapprox(norm(qtn), 1.0)
            @test maximum(last, size(TensorNetwork(qtn))) <= χ
        end
    end

    @testset "Canonization" begin
        using Tenet

        @testset "contract" begin
            qtn = rand(Chain, Open, State; n = 5, p = 2, χ = 20)
            canonized = canonize(qtn)

            @test_throws ArgumentError contract(canonized, :between, Site(1), Site(2); direction = :dummy)
            @test_throws ArgumentError contract!(canonized, :between, Site(1), Site(2); direction = :dummy)

            canonized = canonize(qtn)

            for i in 1:4
                contract_some = contract(canonized, :between, Site(i), Site(i + 1))
                Bᵢ = select(contract_some, :tensor, Site(i))

                @test isapprox(contract(TensorNetwork(contract_some)), contract(TensorNetwork(qtn)))

                @test length(tensors(canonized)) == length(tensors(contract_some)) + 1 # We removed one singular value vector
                @test_throws ArgumentError select(contract_some, :between, Site(i), Site(i + 1))

                @test isrightcanonical(contract_some, Site(i))
                @test isleftcanonical(
                    contract(canonized, :between, Site(i), Site(i + 1); direction = :right),
                    Site(i + 1),
                )

                Γᵢ = select(canonized, :tensor, Site(i))
                Λᵢ₊₁ = select(canonized, :between, Site(i), Site(i + 1))
                @test Bᵢ ≈ contract(Γᵢ, Λᵢ₊₁; dims = ())
            end
        end

        @testset "canonize_site" begin
            qtn = Chain(State(), Open(), [rand(4, 4), rand(4, 4, 4), rand(4, 4)])

            @test_throws ArgumentError canonize_site!(qtn, Site(1); direction = :left)
            @test_throws ArgumentError canonize_site!(qtn, Site(3); direction = :right)

            for method in [:qr, :svd]
                canonized = canonize_site(qtn, site"1"; direction = :right, method = method)
                @test isleftcanonical(canonized, site"1")
                @test isapprox(
                    contract(transform(TensorNetwork(canonized), Tenet.HyperindConverter())),
                    contract(TensorNetwork(qtn)),
                )

                canonized = canonize_site(qtn, site"2"; direction = :right, method = method)
                @test isleftcanonical(canonized, site"2")
                @test isapprox(
                    contract(transform(TensorNetwork(canonized), Tenet.HyperindConverter())),
                    contract(TensorNetwork(qtn)),
                )

                canonized = canonize_site(qtn, site"2"; direction = :left, method = method)
                @test isrightcanonical(canonized, site"2")
                @test isapprox(
                    contract(transform(TensorNetwork(canonized), Tenet.HyperindConverter())),
                    contract(TensorNetwork(qtn)),
                )

                canonized = canonize_site(qtn, site"3"; direction = :left, method = method)
                @test isrightcanonical(canonized, site"3")
                @test isapprox(
                    contract(transform(TensorNetwork(canonized), Tenet.HyperindConverter())),
                    contract(TensorNetwork(qtn)),
                )
            end

            # Ensure that svd creates a new tensor
            @test length(tensors(canonize_site(qtn, Site(2); direction = :left, method = :svd))) == 4
        end

        @testset "canonize" begin
            using Qrochet: isleftcanonical, isrightcanonical

            qtn = MPS([rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])
            canonized = canonize(qtn)

            @test length(tensors(canonized)) == 9 # 5 tensors + 4 singular values vectors
            @test isapprox(
                contract(transform(TensorNetwork(canonized), Tenet.HyperindConverter())),
                contract(TensorNetwork(qtn)),
            )
            @test isapprox(norm(qtn), norm(canonized))

            # Extract the singular values between each adjacent pair of sites in the canonized chain
            Λ = [select(canonized, :between, Site(i), Site(i + 1)) for i in 1:4]
            @test map(λ -> sum(abs2, λ), Λ) ≈ ones(length(Λ)) * norm(canonized)^2

            for i in 1:5
                canonized = canonize(qtn)

                if i == 1
                    @test isleftcanonical(canonized, Site(i))
                elseif i == 5 # in the limits of the chain, we get the norm of the state
                    contract!(canonized, :between, Site(i - 1), Site(i); direction = :right)
                    tensor = select(canonized, :tensor, Site(i))
                    replace!(TensorNetwork(canonized), tensor => tensor / norm(canonized))
                    @test isleftcanonical(canonized, Site(i))
                else
                    contract!(canonized, :between, Site(i - 1), Site(i); direction = :right)
                    @test isleftcanonical(canonized, Site(i))
                end
            end

            for i in 1:5
                canonized = canonize(qtn)

                if i == 1 # in the limits of the chain, we get the norm of the state
                    contract!(canonized, :between, Site(i), Site(i + 1); direction = :left)
                    tensor = select(canonized, :tensor, Site(i))
                    replace!(TensorNetwork(canonized), tensor => tensor / norm(canonized))
                    @test isrightcanonical(canonized, Site(i))
                elseif i == 5
                    @test isrightcanonical(canonized, Site(i))
                else
                    contract!(canonized, :between, Site(i), Site(i + 1); direction = :left)
                    @test isrightcanonical(canonized, Site(i))
                end
            end
        end

        @testset "mixed_canonize" begin
            qtn = Chain(State(), Open(), [rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])
            canonized = mixed_canonize(qtn, Site(3))

            @test isleftcanonical(canonized, Site(1))
            @test isleftcanonical(canonized, Site(2))
            @test !isleftcanonical(canonized, Site(3)) && !isrightcanonical(canonized, Site(3))
            @test isrightcanonical(canonized, Site(4))
            @test isrightcanonical(canonized, Site(5))

            @test isapprox(
                contract(transform(TensorNetwork(canonized), Tenet.HyperindConverter())),
                contract(TensorNetwork(qtn)),
            )
        end
    end

    # TODO test `evolve!` methods
end
