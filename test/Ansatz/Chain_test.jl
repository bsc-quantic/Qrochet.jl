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

    @testset "canonize" begin
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
                    @test is_right_canonical(canonize(qtn, Site(i); direction=:right, mode=mode), Site(i))
                elseif i != length(sites(qtn))
                    @test is_left_canonical(canonize(qtn, Site(i); direction=:left, mode=mode), Site(i))
                end
            end
        end
    end
end
