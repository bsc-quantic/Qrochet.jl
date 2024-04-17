using Tenet

const __indexcounter::Threads.Atomic{Int} = Threads.Atomic{Int}(1)

currindex() = Tenet.letter(__indexcounter[])
nextindex() = (__indexcounter.value >= 135000) ? resetindex() : Tenet.letter(Threads.atomic_add!(__indexcounter, 1))
resetindex() = Tenet.letter(Threads.atomic_xchg!(__indexcounter, 1))
