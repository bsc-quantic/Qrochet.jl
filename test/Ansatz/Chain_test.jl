@testset "Chain ansatz" begin
    qtn = Chain(State(), Periodic(), [rand(2, 4, 4) for _ in 1:3])
    @test socket(qtn) == State()
    @test ninputs(qtn) == 0
    @test noutputs(qtn) == 3
    @test issetequal(sites(qtn), [site"1", site"2", site"3"])
    @test boundary(qtn) == Periodic()

    qtn = Chain(State(), Open(), [rand(2, 2), rand(2, 2, 2), rand(2, 2)])
    @test socket(qtn) == State()
    @test ninputs(qtn) == 0
    @test noutputs(qtn) == 3
    @test issetequal(sites(qtn), [site"1", site"2", site"3"])
    @test boundary(qtn) == Open()

    qtn = Chain(Operator(), Periodic(), [rand(2, 2, 4, 4) for _ in 1:3])
    @test socket(qtn) == Operator()
    @test ninputs(qtn) == 3
    @test noutputs(qtn) == 3
    @test issetequal(sites(qtn), [site"1", site"2", site"3", site"1'", site"2'", site"3'"])
    @test boundary(qtn) == Periodic()

    qtn = Chain(Operator(), Open(), [rand(2, 2, 4), rand(2, 2, 4, 4), rand(2, 2, 4)])
    @test socket(qtn) == Operator()
    @test ninputs(qtn) == 3
    @test noutputs(qtn) == 3
    @test issetequal(sites(qtn), [site"1", site"2", site"3", site"1'", site"2'", site"3'"])
    @test boundary(qtn) == Open()

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

        @test_throws ArgumentError leftsite(qtn, Site(1))
        @test_throws ArgumentError rightsite(qtn, Site(3))

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

        # TODO: Uncomment when `select(:between)` is working
        # singular_values = select(qtn, :between, Site(2), Site(3))
        # truncated = Qrochet.truncate(qtn, [Site(2), Site(3)]; threshold = singular_values[2]+0.1)
        # @test size(TensorNetwork(truncated), rightindex(truncated, Site(2))) == 1
        # @test size(TensorNetwork(truncated), leftindex(truncated, Site(3))) == 1
    end

    @testset "rand" begin
        using LinearAlgebra: norm

        chi = 10
        qtn = rand(Chain, Open, State; n = 6, p = 2, χ = chi)
        @test socket(qtn) == State()
        @test ninputs(qtn) == 0
        @test noutputs(qtn) == 6
        @test issetequal(sites(qtn), [site"1", site"2", site"3", site"4", site"5", site"6"])
        @test boundary(qtn) == Open()
        @test isapprox(norm(qtn), 1.0)

        inds_ = reduce(vcat, [inds(tensor) for tensor in tensors(qtn)])
        unique_inds = filter(index -> count(x -> x == index, inds_) == 1, inds_)
        @test !any(index -> size(TensorNetwork(qtn), index) > chi, unique_inds)

        qtn = rand(Chain, Open, Operator; n = 6, p = 2, χ = chi)
        @test socket(qtn) == Operator()
        @test ninputs(qtn) == 6
        @test noutputs(qtn) == 6
        @test issetequal(sites(qtn),
            [site"1", site"2", site"3", site"4", site"5", site"6",
            site"1'", site"2'", site"3'", site"4'", site"5'", site"6'"])
        @test boundary(qtn) == Open()
        @test isapprox(norm(qtn), 1.0)

        inds_ = reduce(vcat, [inds(tensor) for tensor in tensors(qtn)])
        unique_inds = filter(index -> count(x -> x == index, inds_) == 1, inds_)
        @test !any(index -> size(TensorNetwork(qtn), index) > chi, unique_inds)
    end

    @testset "Canonization" begin
        using Tenet

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
end
