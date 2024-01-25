@testset "Site" begin
    s = Site(1)
    @test s.id == 1
    @test s.dual == false

    s = Site(1; dual = true)
    @test s.id == 1
    @test s.dual == true

    s = site"1"
    @test s.id == 1
    @test s.dual == false

    s = site"1'"
    @test s.id == 1
    @test s.dual == true

    s = site"1" |> adjoint
    @test s.id == 1
    @test s.dual == true

    s = site"1'" |> adjoint
    @test s.id == 1
    @test s.dual == false
end
