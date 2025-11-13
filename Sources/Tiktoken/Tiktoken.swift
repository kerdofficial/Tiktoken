import Foundation

public struct Tiktoken {
    
    public static let shared: Tiktoken = .init()
    
    private init() {}
    
    public func getEncoding(_ name: String) async throws -> Encoding? {
        print("Looking up vocab for model: \(name)")
        guard let vocab = Model.getEncoding(name) else {
            print("No vocab found for model: \(name)")
            return nil
        }
        print("Found vocab: \(vocab.name)")
        print("Loading ranks from: \(vocab.url)")
        
        let encoder = await loadRanks(vocab)
        
        if encoder.isEmpty {
            print("Failed to load ranks - encoder is empty!")
            return nil
        }
        
        print("Loaded \(encoder.count) ranks")
        
        let regex = try NSRegularExpression(pattern: vocab.pattern)
        let encoding = Encoding(name: name, regex: regex, mergeableRanks: encoder, specialTokens: vocab.specialTokens)
        return encoding
    }
    
//    public func getEncoding(for vocab: Vocab) -> Encoding? {
//        return nil
//    }
//    
//    public func register() {
//        // TODO: Register model and Encoding
//    }
//    
//    public func clear() {
//        // TODO: Clear all cached encoding
//    }
}

private extension Tiktoken {
    func loadRanks(_ vocab: Vocab) async -> [[UInt8]: Int] {
        if ["gpt2", "gpt3"].contains(vocab.name) {
            return await Load.dataGymToMergeableBpeRanks(vocabBpeFile: vocab.url)
        } else {
            return await Load.loadTiktokenBpe(url: vocab.url)
        }
    }
}
