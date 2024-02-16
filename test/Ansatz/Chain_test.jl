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

    @testset "canonize" begin
        using Tenet

        function is_left_canonical(qtn, s::Site)
           label_r = rightindex(qtn, s)
           A = select(qtn, :tensor, s)
           try
               contracted = contract(A, replace(conj(A), label_r => :new_ind_name))
               return isapprox(contracted, Matrix{Float64}(I, size(A, label_r), size(A, label_r)), atol=1e-12)
           catch
               return false
           end
       end

        function is_right_canonical(qtn, s::Site)
           label_l = leftindex(qtn, s)
           A = select(qtn, :tensor, s)
           try
               contracted = contract(A, replace(conj(A), label_l => :new_ind_name))
               return isapprox(contracted, Matrix{Float64}(I, size(A, label_l), size(A, label_l)), atol=1e-12)
           catch
               return false
           end
       end

        qtn = Chain(State(), Open(), [rand(4, 4), rand(4, 4, 4), rand(4, 4)])

        @test_throws ArgumentError canonize!(qtn, Site(1); direction=:right)
        @test_throws ArgumentError canonize!(qtn, Site(3); direction=:left)

        for mode in [:qr, :svd]
            for i in 1:length(sites(qtn))
                if i != 1
                    canonized = canonize(qtn, Site(i); direction=:right, mode=mode)
                    @test is_right_canonical(canonized, Site(i))
                    @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperindConverter())), contract(TensorNetwork(qtn)))
                elseif i != length(sites(qtn))
                    canonized = canonize(qtn, Site(i); direction=:left, mode=mode)
                    @test is_left_canonical(canonized, Site(i))
                    @test isapprox(contract(transform(TensorNetwork(canonized), Tenet.HyperindConverter())), contract(TensorNetwork(qtn)))
                end
            end
        end

        # Ensure that svd creates a new tensor
        @test length(tensors(canonize(qtn, Site(2); direction=:right, mode=:svd))) == 4
    end
end
