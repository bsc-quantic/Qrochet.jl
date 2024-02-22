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

    @testset "Canonization" begin
        using Tenet

        @testset "canonize_site" begin
            qtn = Chain(State(), Open(), [rand(4, 4), rand(4, 4, 4), rand(4, 4)])

            @test_throws ArgumentError canonize_site!(qtn, Site(1); direction = :right)
            @test_throws ArgumentError canonize_site!(qtn, Site(3); direction = :left)

            for method in [:qr, :svd]
                for i in 1:length(sites(qtn))
                    if i != 1
                        canonized = canonize_site(qtn, Site(i); direction = :right, method = method)
                        @test is_right_canonical(canonized, Site(i))
                        @test isapprox(
                            contract(transform(TensorNetwork(canonized), Tenet.HyperindConverter())),
                            contract(TensorNetwork(qtn)),
                        )
                    elseif i != length(sites(qtn))
                        canonized = canonize_site(qtn, Site(i); direction = :left, method = method)
                        @test is_left_canonical(canonized, Site(i))
                        @test isapprox(
                            contract(transform(TensorNetwork(canonized), Tenet.HyperindConverter())),
                            contract(TensorNetwork(qtn)),
                        )
                    end
                end
            end

            # Ensure that svd creates a new tensor
            @test length(tensors(canonize_site(qtn, Site(2); direction = :right, method = :svd))) == 4
        end

        @testset "mixed_canonize" begin
            qtn = Chain(State(), Open(), [rand(4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4, 4), rand(4, 4)])
            canonized = mixed_canonize(qtn, Site(3))

            @test is_left_canonical(canonized, Site(1))
            @test is_left_canonical(canonized, Site(2))
            @test is_left_canonical(canonized, Site(3))
            @test is_right_canonical(canonized, Site(4))
            @test is_right_canonical(canonized, Site(5))

            @test length(tensors(canonized)) == 6 # 5 tensors + 1 singular value matrix

            @test isapprox(
                contract(transform(TensorNetwork(canonized), Tenet.HyperindConverter())),
                contract(TensorNetwork(qtn)),
            )
        end
    end
end
